#pragma once

#include <nanospline/PatchBase.h>

namespace nanospline {

template <typename _Scalar, int _dim>
class OffsetPatch final : public PatchBase<_Scalar, _dim>
{
public:
    static_assert(_dim == 3, "Offset surface must be 3D only.");

    using Base = PatchBase<_Scalar, _dim>;
    using ThisType = OffsetPatch<_Scalar, _dim>;
    using Scalar = typename Base::Scalar;
    using Point = typename Base::Point;
    using UVPoint = typename Base::UVPoint;
    using Base::inverse_evaluate;

public:
    OffsetPatch() = default;

    std::unique_ptr<Base> clone() const override
    {
        auto patch = std::make_unique<ThisType>();
        patch->set_base_surface(get_base_surface());
        patch->set_offset(m_offset);
        patch->set_u_lower_bound(get_u_lower_bound());
        patch->set_u_upper_bound(get_u_upper_bound());
        patch->set_v_lower_bound(get_v_lower_bound());
        patch->set_v_upper_bound(get_v_upper_bound());
        patch->initialize();
        return patch;
    }

    PatchEnum get_patch_type() const override { return PatchEnum::OFFSET; }

    const Base* get_base_surface() const { return m_base_surface.get(); }
    void set_base_surface(const Base* base_surface)
    {
        if (base_surface != nullptr) {
            m_base_surface = base_surface->clone();
        } else {
            m_base_surface = nullptr;
        }
    }
    void set_base_surface(std::unique_ptr<Base>&& base_surface) { m_base_surface = base_surface; }

    Scalar get_offset() const { return m_offset; }
    void set_offset(Scalar offset) { m_offset = offset; }

public:
    Point evaluate(Scalar u, Scalar v) const override
    {
        assert_valid_base_surface();
        const auto p = m_base_surface->evaluate(u, v);
        const auto du = m_base_surface->evaluate_derivative_u(u, v);
        const auto dv = m_base_surface->evaluate_derivative_v(u, v);
        const auto n = du.cross(dv).normalized();
        if (m_offset != 0 && !n.array().isFinite().all()) {
            std::cerr << "Warning: evaluating offset surface at singular point: " << u << ", " << v
                      << std::endl;
            return p;
        }

        return p + n * m_offset;
    }

    Point evaluate_derivative_u(Scalar u, Scalar v) const override
    {
        assert_valid_base_surface();
        const auto du = m_base_surface->evaluate_derivative_u(u, v);
        const auto dv = m_base_surface->evaluate_derivative_v(u, v);
        const auto duu = m_base_surface->evaluate_2nd_derivative_uu(u, v);
        const auto duv = m_base_surface->evaluate_2nd_derivative_uv(u, v);
        const auto n = du.cross(dv);
        const auto r = duu.cross(dv) + du.cross(duv);
        const auto n_sq = n.squaredNorm();

        constexpr Scalar TOL = std::numeric_limits<Scalar>::epsilon();
        if (n_sq < TOL) {
            std::cerr << "Warning: evaluating derivative at singular point: " << u << ", " << v
                      << std::endl;
        }

        return du + m_offset * (r / std::sqrt(n_sq) - r.dot(n) * n / (n_sq * std::sqrt(n_sq)));
    }

    Point evaluate_derivative_v(Scalar u, Scalar v) const override
    {
        assert_valid_base_surface();
        const auto du = m_base_surface->evaluate_derivative_u(u, v);
        const auto dv = m_base_surface->evaluate_derivative_v(u, v);
        const auto duv = m_base_surface->evaluate_2nd_derivative_uv(u, v);
        const auto dvv = m_base_surface->evaluate_2nd_derivative_vv(u, v);
        const auto n = du.cross(dv);
        const auto r = duv.cross(dv) + du.cross(dvv);
        const auto n_sq = n.squaredNorm();

        constexpr Scalar TOL = std::numeric_limits<Scalar>::epsilon();
        if (n_sq < TOL) {
            std::cerr << "Warning: evaluating derivative at singular point: " << u << ", " << v
                      << std::endl;
        }

        return dv + m_offset * (r / std::sqrt(n_sq) - r.dot(n) * n / (n_sq * std::sqrt(n_sq)));
    }

    // Computing 2nd derivatives of offset surface analytically requires 3rd
    // derivative informaiton of the base surface, which is not supported by
    // nanospline.  Thus, we approximate it using finite difference.

    Point evaluate_2nd_derivative_uu(Scalar u, Scalar v) const override
    {
        assert_valid_base_surface();
        const Scalar delta_u = (m_u_upper - m_u_lower) * 1e-6;
        const auto du_before = evaluate_derivative_u(u - delta_u, v);
        const auto du_after = evaluate_derivative_u(u + delta_u, v);

        return (du_after - du_before) / (delta_u * 2);
    }

    Point evaluate_2nd_derivative_vv(Scalar u, Scalar v) const override
    {
        assert_valid_base_surface();
        const Scalar delta_v = (m_v_upper - m_v_lower) * 1e-6;
        const auto dv_before = evaluate_derivative_v(u, v - delta_v);
        const auto dv_after = evaluate_derivative_v(u, v + delta_v);

        return (dv_after - dv_before) / (delta_v * 2);
    }

    Point evaluate_2nd_derivative_uv(Scalar u, Scalar v) const override
    {
        assert_valid_base_surface();
        const Scalar delta_v = (m_v_upper - m_v_lower) * 1e-6;
        const auto du_before = evaluate_derivative_u(u, v - delta_v);
        const auto du_after = evaluate_derivative_u(u, v + delta_v);

        return (du_after - du_before) / (delta_v * 2);
    }

    std::tuple<UVPoint, bool> inverse_evaluate(const Point& p,
        const Scalar min_u,
        const Scalar max_u,
        const Scalar min_v,
        const Scalar max_v) const override
    {
        assert_valid_base_surface();

        UVPoint uv;
        bool converged = false;
        std::tie(uv, converged) = m_base_surface->inverse_evaluate(p, min_u, max_u, min_v, max_v);
        if (!converged) {
            return Base::inverse_evaluate(p, uv[0], uv[1], min_u, max_u, min_v, max_v);
        } else {
            std::tie(uv, converged) =
                Base::inverse_evaluate(p, uv[0], uv[1], min_u, max_u, min_v, max_v);
            if (converged) {
                return {uv, converged};
            } else {
                return Base::inverse_evaluate(p, uv[0], uv[1], min_u, max_u, min_v, max_v);
            }
        }
    }

    void initialize() override
    {
        assert_valid_base_surface();
        Base::set_degree_u(m_base_surface->get_degree_u());
        Base::set_degree_v(m_base_surface->get_degree_v());

        constexpr Scalar TOL = std::numeric_limits<Scalar>::epsilon() * 100;
        if (m_base_surface->get_periodic_u()) {
            if (std::abs(m_base_surface->get_u_lower_bound() - m_u_lower) < TOL &&
                std::abs(m_base_surface->get_u_upper_bound() - m_u_upper) < TOL) {
                Base::set_periodic_u(true);
            }
        }
        if (m_base_surface->get_periodic_v()) {
            if (std::abs(m_base_surface->get_v_lower_bound() - m_v_lower) < TOL &&
                std::abs(m_base_surface->get_v_upper_bound() - m_v_upper) < TOL) {
                Base::set_periodic_v(true);
            }
        }
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
        throw not_implemented_error("OffsetPatch patch does not support weight");
    }
    virtual void set_weight(int i, int j, Scalar val) override
    {
        throw not_implemented_error("OffsetPatch patch does not support weight");
    }

    virtual int get_num_knots_u() const override { return 0; }
    virtual Scalar get_knot_u(int i) const override
    {
        throw not_implemented_error("OffsetPatch patch does not support knots");
    }
    virtual void set_knot_u(int i, Scalar val) override
    {
        throw not_implemented_error("OffsetPatch patch does not support knots");
    }

    virtual int get_num_knots_v() const override { return 0; }
    virtual Scalar get_knot_v(int i) const override
    {
        throw not_implemented_error("OffsetPatch patch does not support knots");
    }
    virtual void set_knot_v(int i, Scalar val) override
    {
        throw not_implemented_error("OffsetPatch patch does not support knots");
    }

public:
    int num_control_points_u() const override { return 0; }

    int num_control_points_v() const override { return 0; }

    UVPoint get_control_point_preimage(int i, int j) const override
    {
        throw not_implemented_error("OffsetPatch does not need control points.");
    }

protected:
    int num_recommended_samples_u() const override {
        return m_base_surface->num_recommended_samples_u();
    }

    int num_recommended_samples_v() const override {
        return m_base_surface->num_recommended_samples_v();
    }

private:
    void assert_valid_base_surface() const
    {
        if (m_base_surface == nullptr) {
            throw invalid_setting_error("Invalid base surface!");
        }
    }

private:
    std::unique_ptr<PatchBase<_Scalar, _dim>> m_base_surface;
    Scalar m_offset = 0;
    Scalar m_u_lower = 0;
    Scalar m_u_upper = 1;
    Scalar m_v_lower = 0;
    Scalar m_v_upper = 1;
};

} // namespace nanospline
