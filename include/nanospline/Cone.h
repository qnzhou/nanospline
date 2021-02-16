#pragma once

#include <nanospline/PatchBase.h>

namespace nanospline {

template <typename _Scalar, int _dim>
class Cone final : public PatchBase<_Scalar, _dim>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    static_assert(_dim > 2, "Dimension must be larger than 2.");

    using Base = PatchBase<_Scalar, _dim>;
    using ThisType = Cone<_Scalar, _dim>;
    using Scalar = typename Base::Scalar;
    using Point = typename Base::Point;
    using UVPoint = typename Base::UVPoint;
    using Frame = Eigen::Matrix<Scalar, 3, _dim>;

public:
    Cone()
    {
        m_location.setZero();
        m_frame.setZero();
        m_frame(0, 0) = 1;
        m_frame(1, 1) = 1;
        m_frame(2, 2) = 1;
        Base::set_degree_u(2);
        Base::set_degree_v(1);
    }

    std::unique_ptr<Base> clone() const override { return std::make_unique<ThisType>(*this); }
    PatchEnum get_patch_type() const override { return PatchEnum::CONE; }

    const Point& get_location() const { return m_location; }
    void set_location(const Point& p) { m_location = p; }

    const Frame& get_frame() const { return m_frame; }
    void set_frame(const Frame& f) { m_frame = f; }

    Scalar get_radius() const { return m_radius; }
    void set_radius(Scalar r) { m_radius = r; }

    Scalar get_angle() const { return m_angle; }
    void set_angle(Scalar a) { m_angle = a; }

public:
    Point evaluate(Scalar u, Scalar v) const override
    {
        return m_location +
               (m_radius + v * std::sin(m_angle)) *
                   (m_frame.row(0) * std::cos(u) + m_frame.row(1) * std::sin(u)) +
               m_frame.row(2) * v * std::cos(m_angle);
    }

    Point evaluate_derivative_u(Scalar u, Scalar v) const override
    {
        return (m_radius + v * std::sin(m_angle)) *
               (-m_frame.row(0) * std::sin(u) + m_frame.row(1) * std::cos(u));
    }

    Point evaluate_derivative_v(Scalar u, Scalar v) const override
    {
        return std::sin(m_angle) * (m_frame.row(0) * std::cos(u) + m_frame.row(1) * std::sin(u)) +
               m_frame.row(2) * std::cos(m_angle);
    }

    Point evaluate_2nd_derivative_uu(Scalar u, Scalar v) const override
    {
        return (m_radius + v * std::sin(m_angle)) *
               (-m_frame.row(0) * std::cos(u) - m_frame.row(1) * std::sin(u));
    }

    Point evaluate_2nd_derivative_vv(Scalar u, Scalar v) const override {
        return Point::Zero();
    }

    Point evaluate_2nd_derivative_uv(Scalar u, Scalar v) const override {
        return std::sin(m_angle) *
               (-m_frame.row(0) * std::sin(u) + m_frame.row(1) * std::cos(u));
    }

    UVPoint inverse_evaluate(const Point& p,
        const Scalar min_u,
        const Scalar max_u,
        const Scalar min_v,
        const Scalar max_v) const override
    {
        UVPoint uv;
        const Scalar x = (p - m_location).dot(m_frame.row(0));
        const Scalar y = (p - m_location).dot(m_frame.row(1));
        uv[0] = std::atan2(y, x);

        const Point v = std::cos(uv[0]) * m_frame.row(0) + std::sin(uv[0]) * m_frame.row(1);
        const Point q = m_radius * v;
        const Point d = std::sin(m_angle) * v + std::cos(m_angle) * m_frame.row(2);
        uv[1] = (p - q).dot(d) / d.squaredNorm();
        assert(uv.array().isFinite().all());

        uv[1] = std::max(min_v, std::min(uv[1], max_v));
        if (uv[0] < min_u) {
            int n = static_cast<int>(std::ceil((min_u - uv[0]) / (2 * M_PI)));
            uv[0] += n * 2 * M_PI;
        }
        if (uv[0] > max_u) {
            const Scalar du_min = 2 * M_PI - (uv[0] - min_u);
            const Scalar du_max = uv[0] - max_u;
            if (du_min < du_max) {
                uv[0] = min_u;
            } else {
                uv[0] = max_u;
            }
        }

        assert(uv.array().isFinite().all());
        return uv;
    }

    void initialize() override
    {
        constexpr Scalar TOL = std::numeric_limits<Scalar>::epsilon() * 10;
        assert(std::abs(m_frame.row(0).squaredNorm() - 1) < TOL);
        assert(std::abs(m_frame.row(1).squaredNorm() - 1) < TOL);
        assert(std::abs(m_frame.row(2).squaredNorm() - 1) < TOL);
        assert(std::abs(m_frame.row(0).dot(m_frame.row(1))) < TOL);
        assert(std::abs(m_frame.row(1).dot(m_frame.row(2))) < TOL);
        assert(std::abs(m_frame.row(2).dot(m_frame.row(0))) < TOL);
        assert(m_u_upper > m_u_lower);
        assert(m_v_upper > m_v_lower);
        Base::set_periodic_u((fmod(m_u_upper - m_u_lower, 2 * M_PI) < TOL));
        Base::set_periodic_v(false);
    }

    Scalar get_u_lower_bound() const override { return m_u_lower; }
    Scalar get_u_upper_bound() const override { return m_u_upper; }
    Scalar get_v_lower_bound() const override { return m_v_lower; }
    Scalar get_v_upper_bound() const override { return m_v_upper; }

    void set_u_lower_bound(Scalar t) { m_u_lower = t; }
    void set_u_upper_bound(Scalar t) { m_u_upper = t; }
    void set_v_lower_bound(Scalar t) { m_v_lower = t; }
    void set_v_upper_bound(Scalar t) { m_v_upper = t; }

public:
    virtual int get_num_weights_u() const override { return 0; }
    virtual int get_num_weights_v() const override { return 0; }
    virtual Scalar get_weight(int i, int j) const override
    {
        throw not_implemented_error("Cone patch does not support weight");
    }
    virtual void set_weight(int i, int j, Scalar val) override
    {
        throw not_implemented_error("Cone patch does not support weight");
    }

    virtual int get_num_knots_u() const override { return 0; }
    virtual Scalar get_knot_u(int i) const override
    {
        throw not_implemented_error("Cone patch does not support knots");
    }
    virtual void set_knot_u(int i, Scalar val) override
    {
        throw not_implemented_error("Cone patch does not support knots");
    }

    virtual int get_num_knots_v() const override { return 0; }
    virtual Scalar get_knot_v(int i) const override
    {
        throw not_implemented_error("Cone patch does not support knots");
    }
    virtual void set_knot_v(int i, Scalar val) override
    {
        throw not_implemented_error("Cone patch does not support knots");
    }

public:
    int num_control_points_u() const override { return 0; }

    int num_control_points_v() const override { return 0; }

    UVPoint get_control_point_preimage(int i, int j) const override
    {
        throw not_implemented_error("Cone does not need control points.");
    }

private:
    Point m_location;
    Frame m_frame;
    Scalar m_radius = 1;
    Scalar m_angle = M_PI / 2;
    Scalar m_u_lower = 0;
    Scalar m_u_upper = 2 * M_PI;
    Scalar m_v_lower = 0;
    Scalar m_v_upper = 1;
};

} // namespace nanospline
