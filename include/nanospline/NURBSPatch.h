#pragma once

#include <nanospline/BSplinePatch.h>
#include <nanospline/NURBS.h>
#include <nanospline/PatchBase.h>

namespace nanospline {

template <typename _Scalar, int _dim = 3, int _degree_u = 3, int _degree_v = 3>
class NURBSPatch final : public PatchBase<_Scalar, _dim>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    static_assert(_dim > 0, "Dimension must be positive.");

    using Base = PatchBase<_Scalar, _dim>;
    using Scalar = typename Base::Scalar;
    using UVPoint = typename Base::UVPoint;
    using Point = typename Base::Point;
    using ControlGrid = typename Base::ControlGrid;
    using ThisType = NURBSPatch<_Scalar, _dim, _degree_u, _degree_v>;
    using Weights = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using KnotVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using BSplinePatchHomogeneous = BSplinePatch<Scalar, _dim + 1, _degree_u, _degree_v>;
    using IsoCurveU = NURBS<Scalar, _dim, _degree_u>;
    using IsoCurveV = NURBS<Scalar, _dim, _degree_v>;

public:
    NURBSPatch()
    {
        Base::set_degree_u(_degree_u);
        Base::set_degree_v(_degree_v);
    }

public:
    int num_control_points_u() const override { return m_homogeneous.num_control_points_u(); }

    int num_control_points_v() const override { return m_homogeneous.num_control_points_v(); }

    UVPoint get_control_point_preimage(int i, int j) const override
    {
        return m_homogeneous.get_control_point_preimage(i, j);
    }

    Point get_control_point(int i, int j) const override
    {
        return Base::m_control_grid.row(Base::control_point_linear_index(i, j));
    }

    Point evaluate(Scalar u, Scalar v) const override
    {
        validate_initialization();
        const auto p = m_homogeneous.evaluate(u, v);
        return p.template segment<_dim>(0) / p[_dim];
    }

    Point evaluate_derivative_u(Scalar u, Scalar v) const override
    {
        validate_initialization();
        const auto p = m_homogeneous.evaluate(u, v);
        const auto d = m_homogeneous.evaluate_derivative_u(u, v);

        return (d.template head<_dim>() - p.template head<_dim>() * d[_dim] / p[_dim]) / p[_dim];
    }

    Point evaluate_derivative_v(Scalar u, Scalar v) const override
    {
        validate_initialization();
        const auto p = m_homogeneous.evaluate(u, v);
        const auto d = m_homogeneous.evaluate_derivative_v(u, v);

        return (d.template head<_dim>() - p.template head<_dim>() * d[_dim] / p[_dim]) / p[_dim];
    }

    Point evaluate_2nd_derivative_uu(Scalar u, Scalar v) const override
    {
        validate_initialization();
        constexpr Scalar tol = std::numeric_limits<Scalar>::epsilon();
        const auto p0 = m_homogeneous.evaluate(u, v);
        const auto du = m_homogeneous.evaluate_derivative_u(u, v);
        const auto duu = m_homogeneous.evaluate_2nd_derivative_uu(u, v);

        const Point Auu = duu.template head<_dim>();
        const Scalar wuu = duu[_dim];
        const Scalar w = p0[_dim];
        if (w <= tol) return Point::Zero();

        const Point S = p0.template head<_dim>() / w;
        const Point Su = evaluate_derivative_u(u, v);
        const Scalar wu = du[_dim];

        return (Auu - Su * wu * 2 - S * wuu) / w;
    }

    Point evaluate_2nd_derivative_vv(Scalar u, Scalar v) const override
    {
        validate_initialization();
        constexpr Scalar tol = std::numeric_limits<Scalar>::epsilon();

        const auto p0 = m_homogeneous.evaluate(u, v);
        const auto dv = m_homogeneous.evaluate_derivative_v(u, v);
        const auto dvv = m_homogeneous.evaluate_2nd_derivative_vv(u, v);

        const Point Avv = dvv.template segment<_dim>(0);
        const Scalar wvv = dvv[_dim];
        const Scalar w = p0[_dim];
        if (w <= tol) return Point::Zero();

        const Point S = p0.template segment<_dim>(0) / w;
        const Point Sv = evaluate_derivative_v(u, v);
        const Scalar wv = dv[_dim];

        return (Avv - Sv * wv * 2 - S * wvv) / w;
    }

    Point evaluate_2nd_derivative_uv(Scalar u, Scalar v) const override
    {
        validate_initialization();
        constexpr Scalar tol = std::numeric_limits<Scalar>::epsilon();

        const auto p0 = m_homogeneous.evaluate(u, v);
        const auto du = m_homogeneous.evaluate_derivative_u(u, v);
        const auto dv = m_homogeneous.evaluate_derivative_v(u, v);
        const auto duv = m_homogeneous.evaluate_2nd_derivative_uv(u, v);

        const Point Auv = duv.template segment<_dim>(0);
        const Scalar wuv = duv[_dim];
        const Point S = p0.template segment<_dim>(0) / p0[_dim];
        const Scalar w = p0[_dim];
        if (w <= tol) return Point::Zero();

        const Scalar wu = du[_dim];
        const Scalar wv = dv[_dim];
        const Point Su = evaluate_derivative_u(u, v);
        const Point Sv = evaluate_derivative_v(u, v);

        return (Auv - S * wuv - wu * Sv - wv * Su) / w;
    }

    void initialize() override
    {
        const auto num_control_pts = Base::m_control_grid.rows();
        if (m_weights.size() != num_control_pts) {
            throw invalid_setting_error("Weights and control grid mismatch");
        }

        typename BSplinePatchHomogeneous::ControlGrid ctrl_pts(num_control_pts, _dim + 1);
        ctrl_pts.template leftCols<_dim>() =
            Base::m_control_grid.array().colwise() * m_weights.array();
        ctrl_pts.template rightCols<1>() = m_weights;

        m_homogeneous.set_control_grid(std::move(ctrl_pts));
        m_homogeneous.set_knots_u(m_knots_u);
        m_homogeneous.set_knots_v(m_knots_v);
        m_homogeneous.set_degree_u(Base::get_degree_u());
        m_homogeneous.set_degree_v(Base::get_degree_v());
        m_homogeneous.initialize();
        cache_derivatives(2);
    }

    Scalar get_u_lower_bound() const override { return m_homogeneous.get_u_lower_bound(); }

    Scalar get_v_lower_bound() const override { return m_homogeneous.get_v_lower_bound(); }

    Scalar get_u_upper_bound() const override { return m_homogeneous.get_u_upper_bound(); }

    Scalar get_v_upper_bound() const override { return m_homogeneous.get_v_upper_bound(); }

public:
    const Weights get_weights() const { return m_weights; }

    template <typename Derived>
    void set_weights(const Eigen::PlainObjectBase<Derived>& weights)
    {
        m_weights = weights;
    }

    template <typename Derived>
    void set_weights(Eigen::PlainObjectBase<Derived>&& weights)
    {
        m_weights.swap(weights);
    }

    const KnotVector& get_knots_u() const { return m_knots_u; }

    const KnotVector& get_knots_v() const { return m_knots_v; }

    template <typename Derived>
    void set_knots_u(const Eigen::PlainObjectBase<Derived>& knots)
    {
        m_knots_u = knots;
    }

    template <typename Derived>
    void set_knots_v(const Eigen::PlainObjectBase<Derived>& knots)
    {
        m_knots_v = knots;
    }

    template <typename Derived>
    void set_knots_u(Eigen::PlainObjectBase<Derived>&& knots)
    {
        m_knots_u.swap(knots);
    }

    template <typename Derived>
    void set_knots_v(Eigen::PlainObjectBase<Derived>&& knots)
    {
        m_knots_v.swap(knots);
    }

    IsoCurveU compute_iso_curve_u(Scalar v) const
    {
        auto curve = m_homogeneous.compute_iso_curve_u(v);
        IsoCurveU iso_curve;
        iso_curve.set_homogeneous(curve);
        return iso_curve;
    }

    IsoCurveV compute_iso_curve_v(Scalar u) const
    {
        auto curve = m_homogeneous.compute_iso_curve_v(u);
        IsoCurveV iso_curve;
        iso_curve.set_homogeneous(curve);
        return iso_curve;
    }

    const BSplinePatchHomogeneous& get_homogeneous() const { return m_homogeneous; }

    void set_homogeneous(const BSplinePatchHomogeneous& homogeneous)
    {
        const auto ctrl_pts = homogeneous.get_control_grid();
        m_homogeneous = homogeneous;
        m_weights = ctrl_pts.template rightCols<1>();
        Base::m_control_grid =
            ctrl_pts.template leftCols<_dim>().array().colwise() / m_weights.array();
        m_knots_u = homogeneous.get_knots_u();
        m_knots_v = homogeneous.get_knots_v();
        Base::m_degree_u = homogeneous.get_degree_u();
        Base::m_degree_v = homogeneous.get_degree_v();
        validate_initialization();
    }

    std::vector<ThisType> split_u(Scalar u)
    {
        const auto parts = m_homogeneous.split_u(u);
        std::vector<ThisType> results;
        results.reserve(2);
        for (const auto& c : parts) {
            results.emplace_back();
            results.back().set_homogeneous(c);
        }
        return results;
    }

    std::vector<ThisType> split_v(Scalar v)
    {
        const auto parts = m_homogeneous.split_v(v);
        std::vector<ThisType> results;
        results.reserve(2);
        for (const auto& c : parts) {
            results.emplace_back();
            results.back().set_homogeneous(c);
        }
        return results;
    }

    std::vector<ThisType> split(Scalar u, Scalar v)
    {
        const auto parts = m_homogeneous.split(u, v);
        std::vector<ThisType> results;
        results.reserve(4);
        for (const auto& c : parts) {
            results.emplace_back();
            results.back().set_homogeneous(c);
        }
        return results;
    }

    ThisType subpatch(Scalar u_min, Scalar u_max, Scalar v_min, Scalar v_max) const
    {
        const auto c = m_homogeneous.subpatch(u_min, u_max, v_min, v_max);
        ThisType result;
        result.set_homogeneous(c);
        return result;
    }

    UVPoint approximate_inverse_evaluate(const Point& p,
        const int num_samples,
        const Scalar min_u,
        const Scalar max_u,
        const Scalar min_v,
        const Scalar max_v,
        const int level = 3) const override
    {
        // Only two control points at the endpoints, so finding the closest
        // point doesn't restrict the search at all; default to the parent
        // class function based on sampling where resolution isn't an issue
        if (Base::get_degree_u() < 2 || Base::get_degree_v() < 2) {
            return Base::approximate_inverse_evaluate(p, num_samples, min_u, max_u, min_v, max_v);
        }

        // 1. find closest control point
        // (works for points with 0 weight: Base::m_control_grid is scaled by
        // corresponding weights when initialized, so c_i/w_i = inf as w_i-> 0
        // so small weights push control points "further" from the query // point)
        auto closest_control_pt_index = Base::find_closest_control_point(p);
        int i_min = closest_control_pt_index.first;
        int j_min = closest_control_pt_index.second;

        if (level <= 0) {
            return get_control_point_preimage(i_min, j_min);

        } else {
            // 2. Control points c_{i+/-1,j+/-1} bound the domain containing our
            // desired initial guess; find subdomain corresponding to control
            // point subdomain boundary
            // Conditionals for bound checking
            UVPoint uv_min = get_control_point_preimage(
                i_min > 0 ? i_min - 1 : i_min, j_min > 0 ? j_min - 1 : j_min);
            UVPoint uv_max =
                get_control_point_preimage(i_min < num_control_points_u() - 1 ? i_min + 1 : i_min,
                    j_min < num_control_points_v() - 1 ? j_min + 1 : j_min);

            // 3. split a subcurve to find closest ctrl points on subdomain
            ThisType patch = subpatch(uv_min(0), uv_max(0), uv_min(1), uv_max(1));
            // repeat recursively
            UVPoint uv = patch.approximate_inverse_evaluate(
                p, num_samples, uv_min(0), uv_max(0), uv_min(1), uv_max(1), level - 1);

            // no need to remap coordinates for splines

            return uv;
        }
    }

public:
    void cache_derivatives(int level)
    {
        m_homogeneous.cache_derivatives(level);
    }

    void clear_cache()
    {
        m_homogeneous.clear_cache();
    }

private:
    void validate_initialization() const
    {
        const auto& ctrl_pts = m_homogeneous.get_control_grid();
        if (ctrl_pts.rows() != Base::m_control_grid.rows() || ctrl_pts.rows() != m_weights.rows()) {
            throw invalid_setting_error("NURBS patch is not initialized.");
        }
    }

protected:
    Weights m_weights;
    BSplinePatchHomogeneous m_homogeneous;
    KnotVector m_knots_u;
    KnotVector m_knots_v;
};

} // namespace nanospline
