#pragma once

#include <nanospline/PatchBase.h>

namespace nanospline {

template <typename _Scalar, int _dim>
class Plane final : public PatchBase<_Scalar, _dim>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    static_assert(_dim > 1, "Dimension must be larger than 1.");

    using Base = PatchBase<_Scalar, _dim>;
    using ThisType = Plane<_Scalar, _dim>;
    using Scalar = typename Base::Scalar;
    using Point = typename Base::Point;
    using UVPoint = typename Base::UVPoint;
    using Frame = Eigen::Matrix<Scalar, 2, _dim>;
    using Base::inverse_evaluate;

public:
    Plane()
    {
        m_location.setZero();
        m_frame.setZero();
        m_frame(0, 0) = 1;
        m_frame(1, 1) = 1;
        Base::set_degree_u(1);
        Base::set_degree_v(1);
    }

    std::unique_ptr<Base> clone() const override { return std::make_unique<ThisType>(*this); }
    PatchEnum get_patch_type() const override { return PatchEnum::PLANE; }

    const Point& get_location() const { return m_location; }
    void set_location(const Point& p) { m_location = p; }

    const Frame& get_frame() const { return m_frame; }
    void set_frame(const Frame& f) { m_frame = f; }

public:
    Point evaluate(Scalar u, Scalar v) const override
    {
        return m_location + m_frame.row(0) * u + m_frame.row(1) * v;
    }

    Point evaluate_derivative_u(Scalar u, Scalar v) const override { return m_frame.row(0); }

    Point evaluate_derivative_v(Scalar u, Scalar v) const override { return m_frame.row(1); }

    Point evaluate_2nd_derivative_uu(Scalar u, Scalar v) const override { return Point::Zero(); }

    Point evaluate_2nd_derivative_vv(Scalar u, Scalar v) const override { return Point::Zero(); }

    Point evaluate_2nd_derivative_uv(Scalar u, Scalar v) const override { return Point::Zero(); }

    std::tuple<UVPoint, bool> inverse_evaluate(const Point& p,
        const Scalar min_u,
        const Scalar max_u,
        const Scalar min_v,
        const Scalar max_v) const override
    {
        UVPoint uv;

        uv[0] = (p - m_location).dot(m_frame.row(0)) / m_frame.row(0).squaredNorm();
        uv[1] = (p - m_location).dot(m_frame.row(1)) / m_frame.row(1).squaredNorm();

        uv[0] = std::max(min_u, std::min(uv[0], max_u));
        uv[1] = std::max(min_v, std::min(uv[1], max_v));

        return {uv, true};
    }

    void initialize() override
    {
        constexpr Scalar TOL = std::numeric_limits<Scalar>::epsilon() * 10;
        (void)TOL; // Avoid warning.
        assert(m_frame.row(0).squaredNorm() > TOL);
        assert(m_frame.row(1).squaredNorm() > TOL);
        assert(m_u_upper >= m_u_lower);
        assert(m_v_upper >= m_v_lower);
        Base::set_periodic_u(false);
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
        throw not_implemented_error("Plane patch does not support weight");
    }
    virtual void set_weight(int i, int j, Scalar val) override
    {
        throw not_implemented_error("Plane patch does not support weight");
    }

    virtual int get_num_knots_u() const override { return 0; }
    virtual Scalar get_knot_u(int i) const override
    {
        throw not_implemented_error("Plane patch does not support knots");
    }
    virtual void set_knot_u(int i, Scalar val) override
    {
        throw not_implemented_error("Plane patch does not support knots");
    }

    virtual int get_num_knots_v() const override { return 0; }
    virtual Scalar get_knot_v(int i) const override
    {
        throw not_implemented_error("Plane patch does not support knots");
    }
    virtual void set_knot_v(int i, Scalar val) override
    {
        throw not_implemented_error("Plane patch does not support knots");
    }

public:
    int num_control_points_u() const override { return 0; }

    int num_control_points_v() const override { return 0; }

    UVPoint get_control_point_preimage(int i, int j) const override
    {
        throw not_implemented_error("Plane does not need control points.");
    }

private:
    Point m_location;
    Frame m_frame;
    Scalar m_u_lower = 0;
    Scalar m_u_upper = 1;
    Scalar m_v_lower = 0;
    Scalar m_v_upper = 1;
};

} // namespace nanospline
