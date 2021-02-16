#pragma once

#include <nanospline/PatchBase.h>

namespace nanospline {

template <typename _Scalar, int _dim>
class ExtrusionPatch final : public PatchBase<_Scalar, _dim>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    static_assert(_dim == 3, "Surface of Extrusion is 3D only.");

    using Base = PatchBase<_Scalar, _dim>;
    using ThisType = ExtrusionPatch<_Scalar, _dim>;
    using Scalar = typename Base::Scalar;
    using Point = typename Base::Point;
    using UVPoint = typename Base::UVPoint;
    using Frame = Eigen::Matrix<Scalar, 3, _dim>;
    using ProfileType = CurveBase<_Scalar, _dim>;

public:
    ExtrusionPatch()
    {
        m_location.setZero();
        m_frame.setZero();
        m_frame(0, 0) = 1;
        m_frame(1, 1) = 1;
        m_frame(2, 2) = 1;
        Base::set_degree_u(2);
        Base::set_degree_v(2);
    }

    std::unique_ptr<Base> clone() const override { return std::make_unique<ThisType>(*this); }
    PatchEnum get_patch_type() const override { return PatchEnum::EXTRUSION; }

    const Point& get_location() const { return m_location; }
    void set_location(const Point& p) { m_location = p; }

    const Frame& get_frame() const { return m_frame; }
    void set_frame(const Frame& f) { m_frame = f; }

    const ProfileType* get_profile() const { return m_profile; }
    void set_profile(const ProfileType* profile) { m_profile = profile; }

public:
    Point evaluate(Scalar u, Scalar v) const override
    {
        assert_valid_profile();
        const auto p = m_profile->evaluate(u);
        return m_location + p + m_frame.row(2) * v;
    }

    Point evaluate_derivative_u(Scalar u, Scalar v) const override
    {
        assert_valid_profile();
        return m_profile->evaluate_derivative(u);
    }

    Point evaluate_derivative_v(Scalar u, Scalar v) const override
    {
        assert_valid_profile();
        return m_frame.row(2);
    }

    Point evaluate_2nd_derivative_uu(Scalar u, Scalar v) const override
    {
        assert_valid_profile();
        return m_profile->evaluate_2nd_derivative(u);
    }

    Point evaluate_2nd_derivative_vv(Scalar u, Scalar v) const override
    {
        assert_valid_profile();
        return Point::Zero();
    }

    Point evaluate_2nd_derivative_uv(Scalar u, Scalar v) const override
    {
        assert_valid_profile();
        return Point::Zero();
    }

    UVPoint inverse_evaluate(const Point& p,
        const Scalar min_u,
        const Scalar max_u,
        const Scalar min_v,
        const Scalar max_v) const override
    {
        assert_valid_profile();
        UVPoint uv;
        uv[1] = (p - m_location).dot(m_frame.row(2));
        uv[1] = std::min(max_v, std::max(min_v, uv[1]));
        uv[0] = m_profile->approximate_inverse_evaluate(p - m_frame.row(2) * uv[1], min_u, max_u);

        assert(uv[0] >= min_u && uv[0] <= max_u);
        assert(uv[1] >= min_v && uv[1] <= max_v);
        return uv;
    }

    void initialize() override
    {
        constexpr Scalar TOL = std::numeric_limits<Scalar>::epsilon() * 10;
        assert_valid_profile();
        assert(std::abs(m_frame.row(0).squaredNorm() - 1) < TOL);
        assert(std::abs(m_frame.row(1).squaredNorm() - 1) < TOL);
        assert(std::abs(m_frame.row(2).squaredNorm() - 1) < TOL);
        assert(std::abs(m_frame.row(0).dot(m_frame.row(1))) < TOL);
        assert(std::abs(m_frame.row(1).dot(m_frame.row(2))) < TOL);
        assert(std::abs(m_frame.row(2).dot(m_frame.row(0))) < TOL);
        assert(m_u_upper > m_u_lower);
        assert(m_v_upper > m_v_lower);

        Base::set_periodic_u(m_profile->get_periodic());
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
        throw not_implemented_error("ExtrusionPatch patch does not support weight");
    }
    virtual void set_weight(int i, int j, Scalar val) override
    {
        throw not_implemented_error("ExtrusionPatch patch does not support weight");
    }

    virtual int get_num_knots_u() const override { return 0; }
    virtual Scalar get_knot_u(int i) const override
    {
        throw not_implemented_error("ExtrusionPatch patch does not support knots");
    }
    virtual void set_knot_u(int i, Scalar val) override
    {
        throw not_implemented_error("ExtrusionPatch patch does not support knots");
    }

    virtual int get_num_knots_v() const override { return 0; }
    virtual Scalar get_knot_v(int i) const override
    {
        throw not_implemented_error("ExtrusionPatch patch does not support knots");
    }
    virtual void set_knot_v(int i, Scalar val) override
    {
        throw not_implemented_error("ExtrusionPatch patch does not support knots");
    }

public:
    int num_control_points_u() const override { return 0; }

    int num_control_points_v() const override { return 0; }

    UVPoint get_control_point_preimage(int i, int j) const override
    {
        throw not_implemented_error("ExtrusionPatch does not need control points.");
    }

private:
    void assert_valid_profile() const
    {
        if (m_profile == nullptr) {
            throw invalid_setting_error("Profile not set!");
        }
    }

private:
    Point m_location;
    Frame m_frame;
    const ProfileType* m_profile = nullptr;
    Scalar m_u_lower = 0;
    Scalar m_u_upper = 1;
    Scalar m_v_lower = 0;
    Scalar m_v_upper = 1;
};

} // namespace nanospline