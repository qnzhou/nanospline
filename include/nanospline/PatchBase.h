#pragma once

#include <nanospline/Exceptions.h>
#include <nanospline/enums.h>

#include <Eigen/Core>

#include <cassert>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace nanospline {

template <typename _Scalar, int _dim>
class PatchBase
{
public:
    using Scalar = _Scalar;
    using Point = Eigen::Matrix<Scalar, 1, _dim>;
    using UVPoint = Eigen::Matrix<Scalar, 1, 2>;
    using ControlGrid = Eigen::Matrix<Scalar, Eigen::Dynamic, _dim>;
    using ThisType = PatchBase<_Scalar, _dim>;

public:
    virtual ~PatchBase() = default;
    virtual std::unique_ptr<PatchBase> clone() const = 0;
    virtual PatchEnum get_patch_type() const = 0;

public:
    static constexpr int get_dim() { return _dim; }
    virtual void initialize() = 0;
    virtual Scalar get_u_lower_bound() const = 0;
    virtual Scalar get_u_upper_bound() const = 0;
    virtual Scalar get_v_lower_bound() const = 0;
    virtual Scalar get_v_upper_bound() const = 0;
    virtual int num_control_points_u() const = 0;
    virtual int num_control_points_v() const = 0;
    virtual UVPoint get_control_point_preimage(int i, int j) const = 0;

    bool in_domain_u(Scalar u) const
    {
        constexpr Scalar eps = std::numeric_limits<Scalar>::epsilon();
        const Scalar u_min = get_u_lower_bound();
        const Scalar u_max = get_u_upper_bound();
        return (u >= u_min - eps) && (u <= u_max + eps);
    }

    bool in_domain_v(Scalar v) const
    {
        constexpr Scalar eps = std::numeric_limits<Scalar>::epsilon();
        const Scalar v_min = get_v_lower_bound();
        const Scalar v_max = get_v_upper_bound();
        return (v >= v_min - eps) && (v <= v_max + eps);
    }

    bool is_endpoint_u(Scalar u) const
    {
        return u == get_u_lower_bound() || u == get_u_upper_bound();
    }

    bool is_endpoint_v(Scalar v) const
    {
        return v == get_v_lower_bound() || v == get_v_upper_bound();
    }

    bool in_domain(Scalar u, Scalar v) const { return in_domain_u(u) && in_domain_v(v); }

    void set_degree_u(int degree) { m_degree_u = degree; }

    void set_degree_v(int degree) { m_degree_v = degree; }

    int get_degree_u() const
    {
        assert(m_degree_u >= 0);
        return m_degree_u;
    }

    int get_degree_v() const
    {
        assert(m_degree_v >= 0);
        return m_degree_v;
    }

    const ControlGrid& get_control_grid() const { return m_control_grid; }

    /**
     * Control grid presents a 2D grid of control points.  It is linearized
     * in V-major.  I.e. Let C_uv denotes the control point at (u, v), then
     * control grid = [C_00, C_01, ... C_0n,
     *                 C_10, C_11, ... C_1n,
     *                 ...
     *                 C_m0, C_m1, ... C_mn].
     */
    template <typename Derived>
    void set_control_grid(const Eigen::PlainObjectBase<Derived>& ctrl_grid)
    {
        m_control_grid = ctrl_grid;
    }

    template <typename Derived>
    void set_control_grid(Eigen::PlainObjectBase<Derived>&& ctrl_grid)
    {
        m_control_grid.swap(ctrl_grid);
    }

    template <typename Derived>
    void swap_control_grid(Eigen::PlainObjectBase<Derived>& ctrl_grid)
    {
        m_control_grid.swap(ctrl_grid);
    }

    bool get_periodic_u() const { return m_periodic_u; }
    bool get_periodic_v() const { return m_periodic_v; }
    void set_periodic_u(bool val) { m_periodic_u = val; }
    void set_periodic_v(bool val) { m_periodic_v = val; }

public:
    virtual Point evaluate(Scalar u, Scalar v) const = 0;
    virtual Point evaluate_derivative_u(Scalar u, Scalar v) const = 0;
    virtual Point evaluate_derivative_v(Scalar u, Scalar v) const = 0;
    virtual Point evaluate_2nd_derivative_uu(Scalar u, Scalar v) const = 0;
    virtual Point evaluate_2nd_derivative_vv(Scalar u, Scalar v) const = 0;
    virtual Point evaluate_2nd_derivative_uv(Scalar u, Scalar v) const = 0;

    /**
     * Inverse evaluate. i.e. finding the closest point on patch to a given
     * query point.
     *
     * @param[in] p       The query point.
     * @param[in] min_u   The lower bound on u.
     * @param[in] max_u   The upper bound on u.
     * @param[in] min_v   The lower bound on v.
     * @param[in] max_v   The upper bound on v.
     *
     * @returns  A tuple that contains:
     *           * uv         The uv corresponding to the closest point to p.
     *           * converged  Whether Newton-Raphson has converged.
     *
     * @note This algorithm may not work well if the search region
     * [min_u, max_u] x [min_v, max_v] contains singular points.
     */
    virtual std::tuple<UVPoint, bool> inverse_evaluate(const Point& p,
        const Scalar min_u,
        const Scalar max_u,
        const Scalar min_v,
        const Scalar max_v) const
    {
        const int num_samples_u = num_control_points_u() + 1;
        const int num_samples_v = num_control_points_v() + 1;
        UVPoint uv = approximate_inverse_evaluate(
            p, num_samples_u, num_samples_v, min_u, max_u, min_v, max_v, 10);
        return inverse_evaluate(p, uv[0], uv[1], min_u, max_u, min_v, max_v);
    }

    /**
     * Inverse evaluate with initial guess.
     *
     * @param[in] p       The query point.
     * @param[in] u       Initial guess for u.
     * @param[in] v       Initial guess for v.
     * @param[in] min_u   The lower bound on u.
     * @param[in] max_u   The upper bound on u.
     * @param[in] min_v   The lower bound on v.
     * @param[in] max_v   The upper bound on v.
     * @param[in] num_iterations  The max number of allowed iterations.
     * @param[in] TOL     Convergence tolerence.
     *
     * @returns  A tuple that contains:
     *           * uv         The uv corresponding to the closest point to p.
     *           * converged  Whether Newton-Raphson has converged.
     *
     * @note This algorithm may not work well if the search region
     * [min_u, max_u] x [min_v, max_v] contains singular points.
     */
    virtual std::tuple<UVPoint, bool> inverse_evaluate(const Point& p,
        const Scalar u,
        const Scalar v,
        const Scalar min_u,
        const Scalar max_u,
        const Scalar min_v,
        const Scalar max_v,
        const int num_iterations = 20,
        const Scalar TOL = std::numeric_limits<Scalar>::epsilon() * 100) const
    {
        UVPoint uv{u, v};
        uv[0] = std::max(min_u, std::min(max_u, uv[0]));
        uv[1] = std::max(min_v, std::min(max_v, uv[1]));
        bool converged = newton_raphson(p, uv, num_iterations, TOL, min_u, max_u, min_v, max_v);
        assert(uv[0] >= min_u && uv[0] <= max_u);
        assert(uv[1] >= min_v && uv[1] <= max_v);
        return {uv, converged};
    }

public:
    int num_control_points() const { return num_control_points_u() * num_control_points_v(); }

    Point get_control_point(int i, int j) const
    {
        return m_control_grid.row(get_linear_index(i, j));
    }

    virtual void set_control_point(int i, int j, const Point& p)
    {
        m_control_grid.row(get_linear_index(i, j)) = p;
    }

    virtual int get_num_weights_u() const = 0;
    virtual int get_num_weights_v() const = 0;
    virtual Scalar get_weight(int i, int j) const = 0;
    virtual void set_weight(int i, int j, Scalar val) = 0;

    virtual int get_num_knots_u() const = 0;
    virtual Scalar get_knot_u(int i) const = 0;
    virtual void set_knot_u(int i, Scalar val) = 0;

    virtual int get_num_knots_v() const = 0;
    virtual Scalar get_knot_v(int i) const = 0;
    virtual void set_knot_v(int i, Scalar val) = 0;

protected:
    virtual UVPoint approximate_inverse_evaluate(const Point& p,
        const int num_samples_u,
        const int num_samples_v,
        const Scalar min_u,
        const Scalar max_u,
        const Scalar min_v,
        const Scalar max_v,
        const int level = 3) const
    {
        UVPoint uv(min_u, min_v), uv_2(min_u, min_v);
        Scalar min_dist = std::numeric_limits<Scalar>::max();
        Scalar min_dist_2 = std::numeric_limits<Scalar>::max();
        for (int i = 0; i <= num_samples_u; i++) {
            const Scalar u =
                (i == num_samples_u) ? max_u : (i * (max_u - min_u) / num_samples_u + min_u);
            for (int j = 0; j <= num_samples_v; j++) {
                const Scalar v =
                    (j == num_samples_v) ? max_v : (j * (max_v - min_v) / num_samples_v + min_v);
                const Point q = this->evaluate(u, v);
                const auto dist = (p - q).squaredNorm();
                if (dist < min_dist) {
                    min_dist_2 = min_dist;
                    uv_2 = uv;
                    min_dist = dist;
                    uv = {u, v};
                } else if (dist < min_dist_2) {
                    min_dist_2 = dist;
                    uv_2 = {u, v};
                }
            }
        }

        auto check_sub_range = [&](Scalar u, Scalar v) {
            const auto delta_u = (max_u - min_u) / num_samples_u;
            const auto delta_v = (max_v - min_v) / num_samples_v;
            return PatchBase::approximate_inverse_evaluate(p,
                num_samples_u,
                num_samples_v,
                std::max(u - delta_u, min_u),
                std::min(u + delta_u, max_u),
                std::max(v - delta_v, min_v),
                std::min(v + delta_v, max_v),
                level - 1);
        };

        const Scalar u_range = get_u_upper_bound() - get_u_lower_bound();
        const Scalar v_range = get_v_upper_bound() - get_v_lower_bound();
        auto on_opposite_boundary = [&](Scalar u1, Scalar v1, Scalar u2, Scalar v2) {
            constexpr Scalar TOL = std::numeric_limits<Scalar>::epsilon() * 10;
            bool r = false;
            r |= std::abs(std::abs(u1 - u2) - u_range) < TOL;
            r |= std::abs(std::abs(v1 - v2) - v_range) < TOL;
            return r;
        };

        if (level <= 0) {
            return uv;
        } else if (std::abs(min_dist - min_dist_2) > min_dist * 1e-3 ||
                   !on_opposite_boundary(uv[0], uv[1], uv_2[0], uv_2[1])) {
            return check_sub_range(uv[0], uv[1]);
        } else {
            // Handle the case where two uv points are nearly equally close to
            // the query point.  This is likely due to periodic UV domain.
            // TODO: the unlucky case where more than two uv points are all
            // equally close to the query point is not handled...
            const auto c1 = check_sub_range(uv[0], uv[1]);
            const auto c2 = check_sub_range(uv_2[0], uv_2[1]);
            const auto d1 = (this->evaluate(c1[0], c1[1]) - p).squaredNorm();
            const auto d2 = (this->evaluate(c2[0], c2[1]) - p).squaredNorm();
            return d1 < d2 ? c1 : c2;
        }
    }

    bool newton_raphson(const Point& p,
        UVPoint& uv,
        const int num_iterations,
        const Scalar tol,
        const Scalar min_u,
        const Scalar max_u,
        const Scalar min_v,
        const Scalar max_v) const
    {
        Scalar u = uv[0];
        Scalar v = uv[1];
        UVPoint prev_uv = uv;
        Scalar prev_dist = std::numeric_limits<Scalar>::max();
        for (int i = 0; i < num_iterations; i++) {
            const Point r = this->evaluate(u, v) - p;
            const Scalar dist = r.norm();
            if (dist < tol || std::abs(dist - prev_dist) < tol) {
                break;
            }

            if (dist > prev_dist) {
                // Ops, Newton Raphson diverged...
                // Use the best result so far.
                uv = prev_uv;
                return false;
            }
            prev_dist = dist;
            prev_uv = {u, v};

            const Point Su = this->evaluate_derivative_u(u, v);
            const Point Sv = this->evaluate_derivative_v(u, v);

            if (std::abs(Su.dot(r)) < tol && std::abs(Sv.dot(r)) < tol) {
                // Optimality condition met.  Converged.
                break;
            }

            const Point Suu = this->evaluate_2nd_derivative_uu(u, v);
            const Point Svv = this->evaluate_2nd_derivative_vv(u, v);
            const Point Suv = this->evaluate_2nd_derivative_uv(u, v);

            Eigen::Matrix<Scalar, 2, 2> J;
            J << Su.squaredNorm() + r.dot(Suu), Su.dot(Sv) + r.dot(Suv), Sv.dot(Su) + r.dot(Suv),
                Sv.squaredNorm() + r.dot(Svv);
            if (J.determinant() <= 0) {
                uv = prev_uv;
                return false;
            }
            Eigen::Matrix<Scalar, 2, 1> kappa;
            kappa << -r.dot(Su), -r.dot(Sv);

            Eigen::Matrix<Scalar, 2, 1> delta = J.inverse() * kappa;

            u += delta[0];
            v += delta[1];

            if (u < min_u) u = min_u;
            if (u > max_u) u = max_u;
            if (v < min_v) v = min_v;
            if (v > max_v) v = max_v;
        }
        uv = {u, v};
        return true;
    }

    std::pair<int, int> find_closest_control_point(Point p) const
    {
        Scalar min_dist = std::numeric_limits<Scalar>::max();
        int i_min = 0;
        int j_min = 0;
        for (int ui = 0; ui < num_control_points_u(); ui++) {
            for (int vj = 0; vj < num_control_points_v(); vj++) {
                Point control_point = get_control_point(ui, vj);
                const auto dist = (p - control_point).squaredNorm();
                if (dist < min_dist) {
                    min_dist = dist;
                    i_min = ui;
                    j_min = vj;
                }
            }
        }
        return std::pair<int, int>(i_min, j_min);
    }

    // Translate (i,j) control point indexing into a linear index. Note that
    // control points are ordered in v-major order
    int get_linear_index(int i, int j) const { return i * (num_control_points_v()) + j; }

    Scalar unwrap_parameter_u(Scalar u) const
    {
        assert(m_periodic_u);
        const Scalar u_min = get_u_lower_bound();
        const Scalar u_max = get_u_upper_bound();
        const Scalar period = u_max - u_min;
        u = fmod(u - u_min, period) + u_min;
        if (u < u_min) u += period;
        assert(u >= u_min && u <= u_max);
        return u;
    }

    Scalar unwrap_parameter_v(Scalar v) const
    {
        assert(m_periodic_v);
        const Scalar v_min = get_v_lower_bound();
        const Scalar v_max = get_v_upper_bound();
        const Scalar period = v_max - v_min;
        v = fmod(v - v_min, period) + v_min;
        if (v < v_min) v += period;
        assert(v >= v_min && v <= v_max);
        return v;
    }

    /**
     * Clip parameter t into the range [t_min, t_max] assuming t is periodic
     * with `period` as period.
     */
    Scalar clip_periodic_parameter(Scalar t, Scalar t_min, Scalar t_max, Scalar period) const
    {
        if (t < t_min) {
            int n = static_cast<int>(std::ceil((t_min - t) / period));
            t += n * period;
        } else {
            t = t_min + std::fmod(t - t_min, period);
        }
        assert(t >= t_min);

        if (t > t_max) {
            const Scalar dist_to_min = period - (t - t_min);
            const Scalar dist_to_max = t - t_max;
            t = (dist_to_min < dist_to_max) ? t_min : t_max;
        }
        assert(t <= t_max);
        return t;
    }

protected:
    int m_degree_u = -1;
    int m_degree_v = -1;
    bool m_periodic_u = false;
    bool m_periodic_v = false;
    ControlGrid m_control_grid;
};

} // namespace nanospline
