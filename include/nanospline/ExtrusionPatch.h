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
    using ProfileType = CurveBase<_Scalar, _dim>;

public:
    ExtrusionPatch()
    {
        m_location.setZero();
        m_direction.setZero();
        m_direction[_dim-1] = 1;
        Base::set_degree_u(2);
        Base::set_degree_v(2);
    }

    std::unique_ptr<Base> clone() const override
    {
        auto patch = std::make_unique<ThisType>();
        patch->set_location(get_location());
        patch->set_direction(get_direction());
        patch->set_profile(get_profile());
        patch->set_u_lower_bound(get_u_lower_bound());
        patch->set_u_upper_bound(get_u_upper_bound());
        patch->set_v_lower_bound(get_v_lower_bound());
        patch->set_v_upper_bound(get_v_upper_bound());
        patch->initialize();
        return patch;
    }
    PatchEnum get_patch_type() const override { return PatchEnum::EXTRUSION; }

    const Point& get_location() const { return m_location; }
    void set_location(const Point& p) { m_location = p; }

    const Point& get_direction() const { return m_direction; }
    void set_direction(const Point& d) { m_direction = d; }

    const ProfileType* get_profile() const { return m_profile.get(); }
    void set_profile(const ProfileType* profile)
    {
        if (profile != nullptr) {
            m_profile = profile->clone();
        } else {
            m_profile = nullptr;
        }
    }
    void set_profile(std::unique_ptr<ProfileType>&& profile) { m_profile = std::move(profile); }

public:
    Point evaluate(Scalar u, Scalar v) const override
    {
        assert_valid_profile();
        const auto p = m_profile->evaluate(u);
        return m_location + p + m_direction * v;
    }

    Point evaluate_derivative_u(Scalar u, Scalar v) const override
    {
        assert_valid_profile();
        return m_profile->evaluate_derivative(u);
    }

    Point evaluate_derivative_v(Scalar u, Scalar v) const override
    {
        assert_valid_profile();
        return m_direction;
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
        uv[0] = m_profile->approximate_inverse_evaluate(p-m_location, min_u, max_u);
        uv[1] = (p - m_profile->evaluate(uv[0])).dot(m_direction);
        uv[1] = std::min(max_v, std::max(min_v, uv[1]));

        assert(uv[0] >= min_u && uv[0] <= max_u);
        assert(uv[1] >= min_v && uv[1] <= max_v);
        return uv;
    }

    void initialize() override
    {
        constexpr Scalar TOL = std::numeric_limits<Scalar>::epsilon() * 10;
        (void)TOL; // Avoid warning.
        assert_valid_profile();
        assert(std::abs(m_direction.squaredNorm() - 1) < TOL);
        assert(m_u_upper >= m_u_lower);
        assert(m_v_upper >= m_v_lower);

        if (m_profile->get_periodic()) {
            const auto p0 = m_profile->evaluate(m_u_lower);
            const auto p1 = m_profile->evaluate(m_u_upper);
            Base::set_periodic_u((p1-p0).squaredNorm() < TOL);
        } else {
            Base::set_periodic_u(false);
        }
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
    Point m_direction;
    std::unique_ptr<ProfileType> m_profile = nullptr;
    Scalar m_u_lower = 0;
    Scalar m_u_upper = 1;
    Scalar m_v_lower = 0;
    Scalar m_v_upper = 1;
};

} // namespace nanospline
