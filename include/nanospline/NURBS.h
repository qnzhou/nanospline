#pragma once

#include <Eigen/Core>
#include <iostream>
#include <numeric>

#include <nanospline/BSpline.h>
#include <nanospline/BSplineBase.h>
#include <nanospline/Exceptions.h>
#include <nanospline/RationalBezier.h>

namespace nanospline {

template <typename _Scalar, int _dim = 2, int _degree = 3, bool _generic = (_degree < 0)>
class NURBS : public BSplineBase<_Scalar, _dim, _degree, _generic>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Base = BSplineBase<_Scalar, _dim, _degree, _generic>;
    using Scalar = typename Base::Scalar;
    using Point = typename Base::Point;
    using ControlPoints = typename Base::ControlPoints;
    using WeightVector = Eigen::Matrix<_Scalar, Eigen::Dynamic, 1>;
    using BSplineHomogeneous = BSpline<_Scalar, _dim + 1, _degree, _generic>;

public:
    NURBS() = default;

    explicit NURBS(const std::vector<RationalBezier<_Scalar, _dim, _degree, _generic>>& curves,
        const std::vector<_Scalar>& parameter_bounds)
    {
        combine_rational_Beziers(curves, parameter_bounds);
    }

    explicit NURBS(const std::vector<RationalBezier<_Scalar, _dim, _degree, _generic>>& curves)
    {
        const auto num_curves = curves.size();
        std::vector<_Scalar> parameter_bounds(num_curves + 1);
        std::iota(parameter_bounds.begin(), parameter_bounds.end(), 0);
        combine_rational_Beziers(curves, parameter_bounds);
    }

public:
    Point evaluate(Scalar t) const override
    {
        validate_initialization();
        auto p = m_bspline_homogeneous.evaluate(t);
        return p.template segment<_dim>(0) / p[_dim];
    }

    Scalar inverse_evaluate(const Point& p) const override
    {
        throw not_implemented_error("Too complex, sigh");
    }

    Point evaluate_derivative(Scalar t) const override
    {
        validate_initialization();
        const auto p = m_bspline_homogeneous.evaluate(t);
        const auto d = m_bspline_homogeneous.evaluate_derivative(t);

        return (d.template head<_dim>() - p.template head<_dim>() * d[_dim] / p[_dim]) / p[_dim];
    }

    Point evaluate_2nd_derivative(Scalar t) const override
    {
        const auto p0 = m_bspline_homogeneous.evaluate(t);
        const auto d1 = m_bspline_homogeneous.evaluate_derivative(t);
        const auto d2 = m_bspline_homogeneous.evaluate_2nd_derivative(t);

        const auto c0 = p0.template head<_dim>() / p0[_dim];
        const auto c1 =
            (d1.template head<_dim>() - p0.template head<_dim>() * d1[_dim] / p0[_dim]) / p0[_dim];

        return (d2.template head<_dim>() - d2[_dim] * c0 - 2 * d1[_dim] * c1) / p0[_dim];
    }

    void insert_knot(Scalar t, int multiplicity = 1) override
    {
        validate_initialization();
        m_bspline_homogeneous.insert_knot(t, multiplicity);
        set_homogeneous(m_bspline_homogeneous);
    }

    int remove_knot(Scalar t, int multiplicity = 1, Scalar tol = -1) override
    {
        validate_initialization();
        const auto r = m_bspline_homogeneous.remove_knot(t, multiplicity, tol);
        set_homogeneous(m_bspline_homogeneous);
        return r;
    }

    std::tuple<std::vector<RationalBezier<Scalar, _dim, _degree, _generic>>, std::vector<Scalar>>
    convert_to_RationalBezier() const
    {
        const auto& homogeneous_bspline = get_homogeneous();
        const auto out = homogeneous_bspline.convert_to_Bezier();
        const auto& homogeneous_beziers = std::get<0>(out);
        const auto& parameter_bounds = std::get<1>(out);

        const auto num_segments = homogeneous_beziers.size();

        using CurveType = RationalBezier<Scalar, _dim, _degree, _generic>;
        std::vector<CurveType> segments;
        segments.reserve(num_segments);

        for (size_t i = 0; i < num_segments; i++) {
            const auto& homogeneous_segment = homogeneous_beziers[i];
            CurveType segment;
            segment.set_homogeneous(homogeneous_segment);
            segments.push_back(segment);
        }

        return {segments, parameter_bounds};
    }

    std::vector<Scalar> compute_inflections(
        const Scalar lower, const Scalar upper) const override final
    {
        std::vector<RationalBezier<Scalar, _dim, _degree, _generic>> segments;
        std::vector<Scalar> parameter_bounds;
        std::tie(segments, parameter_bounds) = convert_to_RationalBezier();
        assert(segments.size() + 1 == parameter_bounds.size());

        std::vector<Scalar> res;
        res.reserve(static_cast<size_t>(Base::get_degree()));

        const size_t num_segments = segments.size();
        for (size_t idx = 0; idx < num_segments; idx++) {
            const auto t_min = parameter_bounds[idx];
            const auto t_max = parameter_bounds[idx + 1];
            if (t_max < lower || t_min > upper) {
                continue;
            }

            Scalar normalized_lower = (lower - t_min) / (t_max - t_min);
            Scalar normalized_upper = (upper - t_min) / (t_max - t_min);
            normalized_lower = std::max<Scalar>(normalized_lower, 0);
            normalized_upper = std::min<Scalar>(normalized_upper, 1);

            const auto& curve = segments[idx];
            auto inflections = curve.compute_inflections(normalized_lower, normalized_upper);
            for (auto t : inflections) {
                res.push_back(t * (t_max - t_min) + t_min);
            }
        }

        std::sort(res.begin(), res.end());
        res.erase(std::unique(res.begin(), res.end()), res.end());

        return res;
    }

    std::vector<Scalar> reduce_turning_angle(
        const Scalar lower, const Scalar upper) const override final
    {
        if (_dim != 2) {
            throw std::runtime_error("Turning angle reduction is for 2D curves only");
        }

        std::vector<RationalBezier<Scalar, _dim, _degree, _generic>> segments;
        std::vector<Scalar> parameter_bounds;
        std::tie(segments, parameter_bounds) = convert_to_RationalBezier();
        assert(segments.size() + 1 == parameter_bounds.size());

        std::vector<Scalar> res;
        res.reserve(static_cast<size_t>(Base::get_degree()));

        const size_t num_segments = segments.size();
        for (size_t idx = 0; idx < num_segments; idx++) {
            const auto t_min = parameter_bounds[idx];
            const auto t_max = parameter_bounds[idx + 1];
            if (t_max < lower || t_min > upper) {
                continue;
            }

            Scalar normalized_lower = (lower - t_min) / (t_max - t_min);
            Scalar normalized_upper = (upper - t_min) / (t_max - t_min);
            normalized_lower = std::max<Scalar>(normalized_lower, 0);
            normalized_upper = std::min<Scalar>(normalized_upper, 1);

            const auto& curve = segments[idx];
            auto result = curve.reduce_turning_angle(normalized_lower, normalized_upper);
            for (auto t : result) {
                res.push_back(t * (t_max - t_min) + t_min);
            }
        }

        std::sort(res.begin(), res.end());
        res.erase(std::unique(res.begin(), res.end()), res.end());

        return res;
    }

    std::vector<Scalar> compute_singularities(
        const Scalar lower, const Scalar upper) const override final
    {
        if (_dim != 2) {
            throw std::runtime_error("Singularity computation is for 2D curves only");
        }

        std::vector<RationalBezier<Scalar, _dim, _degree, _generic>> segments;
        std::vector<Scalar> parameter_bounds;
        std::tie(segments, parameter_bounds) = convert_to_RationalBezier();
        assert(segments.size() + 1 == parameter_bounds.size());

        std::vector<Scalar> res;
        res.reserve(static_cast<size_t>(Base::get_degree()));

        const size_t num_segments = segments.size();
        for (size_t idx = 0; idx < num_segments; idx++) {
            const auto t_min = parameter_bounds[idx];
            const auto t_max = parameter_bounds[idx + 1];
            if (t_max < lower || t_min > upper) {
                continue;
            }

            Scalar normalized_lower = (lower - t_min) / (t_max - t_min);
            Scalar normalized_upper = (upper - t_min) / (t_max - t_min);
            normalized_lower = std::max<Scalar>(normalized_lower, 0);
            normalized_upper = std::min<Scalar>(normalized_upper, 1);

            const auto& curve = segments[idx];
            auto result = curve.compute_singularities(normalized_lower, normalized_upper);
            for (auto t : result) {
                res.push_back(t * (t_max - t_min) + t_min);
            }
        }

        std::sort(res.begin(), res.end());
        res.erase(std::unique(res.begin(), res.end()), res.end());

        return res;
    }

    virtual void write(std::ostream& out) const override
    {
        out << "c:\n" << this->m_control_points << "\n";
        out << "k:\n" << this->m_knots << "\n";
        out << "w:\n" << m_weights << "\n";
    }

public:
    void initialize()
    {
        typename BSplineHomogeneous::ControlPoints ctrl_pts(
            Base::m_control_points.rows(), _dim + 1);
        ctrl_pts.template leftCols<_dim>() =
            Base::m_control_points.array().colwise() * m_weights.array();
        ctrl_pts.template rightCols<1>() = m_weights;

        m_bspline_homogeneous.set_control_points(std::move(ctrl_pts));
        m_bspline_homogeneous.set_knots(Base::m_knots);
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
        m_weights.swap(weights);
    }

    const BSplineHomogeneous& get_homogeneous() const { return m_bspline_homogeneous; }

    void set_homogeneous(const BSplineHomogeneous& homogeneous)
    {
        const auto ctrl_pts = homogeneous.get_control_points();
        m_bspline_homogeneous = homogeneous;
        m_weights = ctrl_pts.template rightCols<1>();
        Base::m_control_points =
            ctrl_pts.template leftCols<_dim>().array().colwise() / m_weights.array();
        Base::m_knots = homogeneous.get_knots();
        validate_initialization();
    }

    NURBS < _Scalar, _dim, _degree<0 ? _degree : _degree + 1, _generic> elevate_degree() const
    {
        validate_initialization();
        using TargetType = NURBS < _Scalar, _dim, _degree<0 ? _degree : _degree + 1, _generic>;
        TargetType new_curve;
        new_curve.set_homogeneous(m_bspline_homogeneous.elevate_degree());
        return new_curve;
    }

private:
    void validate_initialization() const
    {
        Base::validate_curve();
        const auto& ctrl_pts = m_bspline_homogeneous.get_control_points();
        if (ctrl_pts.rows() != Base::m_control_points.rows() ||
            ctrl_pts.rows() != m_weights.rows()) {
            throw invalid_setting_error("NURBS curve is not initialized.");
        }
    }

    void combine_rational_Beziers(
        const std::vector<RationalBezier<_Scalar, _dim, _degree, _generic>>& curves,
        const std::vector<_Scalar>& parameter_bounds)
    {
        using CurveType = RationalBezier<_Scalar, _dim, _degree, _generic>;

        const auto num_curves = curves.size();
        std::vector<typename CurveType::BezierHomogeneous> beziers;
        beziers.reserve(num_curves);
        std::for_each(curves.begin(), curves.end(), [&beziers](const CurveType& curve) {
            beziers.push_back(curve.get_homogeneous());
        });
        set_homogeneous(BSplineHomogeneous(beziers, parameter_bounds));
    }

private:
    BSplineHomogeneous m_bspline_homogeneous;
    WeightVector m_weights;
};

} // namespace nanospline
