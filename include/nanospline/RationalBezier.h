#pragma once

#include <Eigen/Core>

#include <nanospline/Bezier.h>
#include <nanospline/BezierBase.h>
#include <nanospline/Exceptions.h>

#if NANOSPLINE_SYMPY
#include <nanospline/internal/auto_inflection_RationalBezier.h>
#include <nanospline/internal/auto_match_tangent_RationalBezier.h>
#include <nanospline/internal/auto_singularity_RationalBezier.h>
#endif

namespace nanospline {

template <typename _Scalar, int _dim = 2, int _degree = 3, bool _generic = (_degree < 0)>
class RationalBezier final : public BezierBase<_Scalar, _dim, _degree, _generic>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Base = BezierBase<_Scalar, _dim, _degree, _generic>;
    using ThisType = RationalBezier<_Scalar, _dim, _degree, _generic>;
    using Scalar = typename Base::Scalar;
    using Point = typename Base::Point;
    using ControlPoints = typename Base::ControlPoints;
    using WeightVector = Eigen::Matrix<_Scalar, _generic ? Eigen::Dynamic : _degree + 1, 1>;
    using BezierHomogeneous = Bezier<_Scalar, _dim + 1, _degree, _generic>;

public:
    CurveEnum get_curve_type() const override { return CurveEnum::RATIONAL_BEZIER; }

    std::unique_ptr<CurveBase<_Scalar, _dim>> clone() const override
    {
        return std::make_unique<ThisType>(*this);
    }

    Point evaluate(Scalar t) const override
    {
        validate_initialization();
        auto p = m_bezier_homogeneous.evaluate(t);
        return p.template head<_dim>() / p[_dim];
    }

    Scalar inverse_evaluate(const Point& p) const override
    {
        throw not_implemented_error("Too complex, sigh");
    }

    Point evaluate_derivative(Scalar t) const override
    {
        validate_initialization();
        const auto p = m_bezier_homogeneous.evaluate(t);
        const auto d = m_bezier_homogeneous.evaluate_derivative(t);

        return (d.template head<_dim>() - p.template head<_dim>() * d[_dim] / p[_dim]) / p[_dim];
    }

    Point evaluate_2nd_derivative(Scalar t) const override
    {
        validate_initialization();
        const auto p0 = m_bezier_homogeneous.evaluate(t);
        const auto d1 = m_bezier_homogeneous.evaluate_derivative(t);
        const auto d2 = m_bezier_homogeneous.evaluate_2nd_derivative(t);

        const auto c0 = p0.template head<_dim>() / p0[_dim];
        const auto c1 =
            (d1.template head<_dim>() - p0.template head<_dim>() * d1[_dim] / p0[_dim]) / p0[_dim];

        return (d2.template head<_dim>() - d2[_dim] * c0 - 2 * d1[_dim] * c1) / p0[_dim];
    }

    virtual void set_control_point(int i, const Point& p) override
    {
        validate_initialization();
        Base::set_control_point(i, p);

        auto q = m_bezier_homogeneous.get_control_point(i);
        q.template segment<_dim>(0) = p * m_weights[i];
        m_bezier_homogeneous.set_control_point(i, q);
    }

    virtual int get_num_weights() const override { return Base::get_num_control_points(); }

    virtual Scalar get_weight(int i) const override { return m_weights[i]; }

    virtual void set_weight(int i, Scalar val) override
    {
        validate_initialization();
        m_weights[i] = val;

        auto q = m_bezier_homogeneous.get_control_point(i);
        q.template segment<_dim>(0) = Base::m_control_points.row(i) * val;
        q[_dim] = val;
        m_bezier_homogeneous.set_control_point(i, q);
    }

    std::vector<Scalar> compute_inflections(const Scalar lower, const Scalar upper) const override
    {
#if NANOSPLINE_SYMPY
        if (_dim != 2) {
            throw std::runtime_error("Inflection computation is for 2D curves only");
        }
        const auto degree = Base::get_degree();
        if (degree <= 2) {
            return {};
        }

        std::vector<Scalar> res;
        try {
            res = internal::compute_RationalBezier_inflections(
                Base::m_control_points, m_weights, lower, upper);
        } catch (infinite_root_error&) {
            res.clear();
        }

        std::sort(res.begin(), res.end());
        res.erase(std::unique(res.begin(), res.end()), res.end());

        return res;
#else
        throw not_implemented_error("This feature require 'NANOSPLINE_SYMPY' compiler flag.");
        return {};
#endif
    }

    Scalar get_turning_angle(Scalar t0, Scalar t1) const override
    {
        if (_dim != 2) {
            throw invalid_setting_error("Turning angle is for 2D only");
        }

        const auto num_ctrl_pts = Base::get_num_control_points();
        Scalar theta = 0;
        Scalar t_prev = t0;
        for (int i = 1; i < num_ctrl_pts; i++) {
            Scalar t_curr = t0 + (Scalar)i / (Scalar)(num_ctrl_pts - 1) * (t1 - t0);
            theta += Base::get_turning_angle(t_prev, t_curr);
            t_prev = t_curr;
        }
        return theta;
    }

    std::vector<Scalar> reduce_turning_angle(const Scalar lower, const Scalar upper) const override
    {
#if NANOSPLINE_SYMPY
        constexpr Scalar tol = static_cast<Scalar>(1e-8);
        if (_dim != 2) {
            throw std::runtime_error("Turning angle reduction is for 2D curves only");
        }
        const auto degree = Base::get_degree();
        if (degree < 2) {
            return {};
        }

        auto tan0 = evaluate_derivative(lower);
        auto tan1 = evaluate_derivative(upper);

        if (tan0.norm() < tol || tan1.norm() < tol || (tan0 + tan1).norm() < 2 * tol) {
            std::vector<Scalar> res;
            res.push_back((lower + upper) / 2);
            return res;
        }

        tan0 = tan0 / tan0.norm();
        tan1 = tan1 / tan1.norm();

        const Eigen::Matrix<Scalar, 2, 1> ave_tangent(
            -(tan0[1] + tan1[1]) / 2, (tan0[0] + tan1[0]) / 2);

        std::vector<Scalar> res;
        try {
            res = nanospline::internal::match_tangent_rational_bezier(
                Base::m_control_points, m_weights, degree, ave_tangent, lower, upper);
        } catch (infinite_root_error&) {
            res.clear();
        }

        std::sort(res.begin(), res.end());
        res.erase(std::unique(res.begin(), res.end()), res.end());

        return res;
#else
        throw not_implemented_error("This feature require 'NANOSPLINE_SYMPY' compiler flag.");
        return {};
#endif
    }

    std::vector<Scalar> compute_singularities(
        const Scalar lower = 0.0, const Scalar upper = 1.0) const override
    {
#if NANOSPLINE_SYMPY
        if (_dim != 2) {
            throw std::runtime_error("Singularity computation is for 2D curves only");
        }

        std::vector<Scalar> res = nanospline::internal::compute_RationalBezier_singularities(
            Base::m_control_points, m_weights, lower, upper);

        std::sort(res.begin(), res.end());
        res.erase(std::unique(res.begin(), res.end()), res.end());

        return res;
#else
        throw not_implemented_error("This feature require 'NANOSPLINE_SYMPY' compiler flag.");
        return {};
#endif
    }

public:
    void initialize() override
    {
        typename BezierHomogeneous::ControlPoints ctrl_pts(Base::m_control_points.rows(), _dim + 1);
        ctrl_pts.template leftCols<_dim>() =
            Base::m_control_points.array().colwise() * m_weights.array();
        ctrl_pts.template rightCols<1>() = m_weights;

        m_bezier_homogeneous.set_control_points(std::move(ctrl_pts));
        m_bezier_homogeneous.set_periodic(Base::get_periodic());
    }

    const WeightVector& get_weights() const { return m_weights; }

    template <typename Derived>
    void set_weights(const Eigen::PlainObjectBase<Derived>& weights)
    {
        m_weights = weights;
    }

    template <typename Derived>
    void set_weights(const Eigen::PlainObjectBase<Derived>&& weights)
    {
        m_weights = std::move(weights);
    }

    const BezierHomogeneous& get_homogeneous() const { return m_bezier_homogeneous; }

    void set_homogeneous(const BezierHomogeneous& homogeneous)
    {
        const auto ctrl_pts = homogeneous.get_control_points();
        m_bezier_homogeneous = homogeneous;
        m_weights = ctrl_pts.template rightCols<1>();
        Base::m_control_points =
            ctrl_pts.template leftCols<_dim>().array().colwise() / m_weights.array();
        Base::set_periodic(homogeneous.get_periodic());
        validate_initialization();
    }

    RationalBezier < _Scalar, _dim,
        _degree<0 ? _degree : _degree + 1, _generic> elevate_degree() const
    {
        validate_initialization();
        using TargetType = RationalBezier < _Scalar, _dim,
              _degree<0 ? _degree : _degree + 1, _generic>;
        TargetType new_curve;
        new_curve.set_homogeneous(m_bezier_homogeneous.elevate_degree());
        assert(new_curve.get_periodic() == Base::get_periodic());
        return new_curve;
    }

private:
    void validate_initialization() const
    {
        const auto& ctrl_pts = m_bezier_homogeneous.get_control_points();
        if (ctrl_pts.rows() != Base::m_control_points.rows() ||
            ctrl_pts.rows() != m_weights.rows()) {
            throw invalid_setting_error("Rational Bezier curve is not initialized.");
        }
        if (m_bezier_homogeneous.get_periodic() != Base::get_periodic()) {
            throw invalid_setting_error("Rational Bezier curve has inconsistent periodicity.");
        }
    }

private:
    BezierHomogeneous m_bezier_homogeneous;
    WeightVector m_weights;
};

} // namespace nanospline
