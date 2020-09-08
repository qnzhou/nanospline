#pragma once

#include <numeric>
#include <vector>

#include <Eigen/Core>

#include <nanospline/BSplineBase.h>
#include <nanospline/Bezier.h>
#include <nanospline/Exceptions.h>

namespace nanospline {

template <typename _Scalar, int _dim = 3, int _degree = 3, bool _generic = (_degree < 0)>
class BSpline : public BSplineBase<_Scalar, _dim, _degree, _generic>
{
public:
    using ThisType = BSpline<_Scalar, _dim, _degree, false>;
    using Base = BSplineBase<_Scalar, _dim, _degree, _generic>;
    using Scalar = typename Base::Scalar;
    using Point = typename Base::Point;
    using ControlPoints = typename Base::ControlPoints;
    using KnotVector = typename Base::KnotVector;
    // Note that BlossomVector has the same length as KnotVector; typedef is
    // just for clarity
    using BlossomVector = typename Base::KnotVector;

public:
    BSpline() = default;

    /**
     * Create a B-Spline by combining a sequence of Bézier curves.
     * These Bézier curves must be at least C0 continous at the joints.
     */
    explicit BSpline(const std::vector<Bezier<_Scalar, _dim, _degree, _generic>>& beziers,
        const std::vector<_Scalar>& parameter_bounds)
    {
        combine_Beziers(beziers, parameter_bounds);
    }

    /**
     * Same as above, except use uniform knot span for each curve.
     */
    explicit BSpline(const std::vector<Bezier<_Scalar, _dim, _degree, _generic>>& beziers)
    {
        const auto num_curves = beziers.size();
        std::vector<_Scalar> parameter_bounds(num_curves + 1);
        std::iota(parameter_bounds.begin(), parameter_bounds.end(), 0);
        combine_Beziers(beziers, parameter_bounds);
    }

public:
    void get_knot_span_control_points(
        int curve_degree, int knot_span, ControlPoints& control_pts) const
    {
        int span_index = knot_span - curve_degree;
        for (int i_control_point = 0; i_control_point <= curve_degree; ++i_control_point) {
            int control_point_index = span_index + i_control_point;
            // verify that there are even enough control points to go around
            assert(control_point_index >= 0);
            assert(control_point_index < static_cast<int>(Base::m_control_points.rows()));
            control_pts.row(i_control_point) = Base::m_control_points.row(control_point_index);
        }
    }

    Point evaluate(Scalar t) const override
    {
        Base::validate_curve();
        const int p = Base::get_degree();
        const int k = Base::locate_span(t);
        assert(p >= 0);
        assert(Base::m_knots.rows() == Base::m_control_points.rows() + p + 1);

        ControlPoints ctrl_pts(p + 1, _dim);
        get_knot_span_control_points(p, k, ctrl_pts);

        return deBoor(t, p, k, ctrl_pts);
    }

    Scalar inverse_evaluate(const Point& p) const override
    {
        throw not_implemented_error("Too complex, sigh");
    }

    void get_derivative_coefficients(
        int curve_degree, int knot_span, ControlPoints& control_pts) const
    {
        int num_control_points = curve_degree + 1; // TODO is this right?
        for (int i = 0; i < num_control_points; i++) {
            int index = i + knot_span + 1;

            const Scalar diff = Base::m_knots[index] - Base::m_knots[index - num_control_points];

            Scalar alpha = 0.0;
            if (diff > 0) {
                alpha = num_control_points / diff;
            }
            control_pts.row(i) = alpha * (control_pts.row(i + 1) - control_pts.row(i));
        }
    }
    Point evaluate_derivative(Scalar t) const override
    {
        Base::validate_curve();
        const int p = Base::get_degree();
        const int k = Base::locate_span(t);
        assert(p >= 0);
        assert(Base::m_knots.rows() == Base::m_control_points.rows() + p + 1);
        assert(Base::m_knots[k] <= t);
        assert(Base::m_knots[k + 1] >= t);

        if (p == 0) return Point::Zero();

        // Only uses the first p rows
        ControlPoints ctrl_pts(p + 1, _dim);
        get_knot_span_control_points(p, k, ctrl_pts);

        int deriv_degree = p - 1;
        get_derivative_coefficients(deriv_degree, k, ctrl_pts);

        return deBoor(t, deriv_degree, k, ctrl_pts);
    }

    Point evaluate_2nd_derivative(Scalar t) const override
    {
        Base::validate_curve();
        const int p = Base::get_degree();
        const int k = Base::locate_span(t);
        assert(p >= 0);
        assert(Base::m_knots.rows() == Base::m_control_points.rows() + p + 1);
        assert(Base::m_knots[k] <= t);
        assert(Base::m_knots[k + 1] >= t);

        if (p <= 1) return Point::Zero();

        // Only uses the first p rows
        ControlPoints ctrl_pts(p + 1, _dim);
        int deriv_degree = p - 1;
        int deriv2_degree = p - 2;

        get_knot_span_control_points(p, k, ctrl_pts);
        get_derivative_coefficients(deriv_degree, k, ctrl_pts);
        get_derivative_coefficients(deriv2_degree, k, ctrl_pts);

        return deBoor(t, deriv2_degree, k, ctrl_pts);
    }

    std::vector<Scalar> compute_inflections(
        const Scalar lower, const Scalar upper) const override final
    {
        using CurveType = Bezier<Scalar, _dim, _degree, _generic>;
        std::vector<CurveType> beziers;
        std::vector<Scalar> parameter_bounds;
        std::tie(beziers, parameter_bounds) = convert_to_Bezier();
        assert(parameter_bounds.size() == beziers.size() + 1);

        std::vector<Scalar> res;
        res.reserve(static_cast<size_t>(Base::get_degree()));

        const size_t num_beziers = beziers.size();
        for (size_t idx = 0; idx < num_beziers; idx++) {
            const auto t_min = parameter_bounds[idx];
            const auto t_max = parameter_bounds[idx + 1];
            if (t_max < lower || t_min > upper) {
                continue;
            }

            Scalar normalized_lower = (lower - t_min) / (t_max - t_min);
            Scalar normalized_upper = (upper - t_min) / (t_max - t_min);
            normalized_lower = std::max<Scalar>(normalized_lower, 0);
            normalized_upper = std::min<Scalar>(normalized_upper, 1);

            const auto& curve = beziers[idx];
            auto inflections = curve.compute_inflections(normalized_lower, normalized_upper);
            for (auto t : inflections) {
                res.push_back(t * (t_max - t_min) + t_min);
            }
        }

        std::sort(res.begin(), res.end());
        res.erase(std::unique(res.begin(), res.end()), res.end());

        return res;
    }

    Scalar get_turning_angle(Scalar t0, Scalar t1) const override
    {
        using CurveType = Bezier<Scalar, _dim, _degree, _generic>;

        if (_dim != 2) {
            throw std::runtime_error("Turning angle reduction is for 2D curves only");
        }

        std::vector<CurveType> beziers;
        std::vector<Scalar> parameter_bounds;
        std::tie(beziers, parameter_bounds) = convert_to_Bezier();

        const auto num_beziers = beziers.size();
        Scalar theta = 0;
        for (size_t i = 0; i < num_beziers; i++) {
            const auto t_min = parameter_bounds[i];
            const auto t_max = parameter_bounds[i + 1];
            if (t_max < t0 || t_min > t1) {
                continue;
            }

            Scalar normalized_lower = (t0 - t_min) / (t_max - t_min);
            Scalar normalized_upper = (t1 - t_min) / (t_max - t_min);
            normalized_lower = std::max<Scalar>(normalized_lower, 0);
            normalized_upper = std::min<Scalar>(normalized_upper, 1);
            theta += beziers[i].get_turning_angle(normalized_lower, normalized_upper);
        }

        return theta;
    }

    std::vector<Scalar> reduce_turning_angle(
        const Scalar lower, const Scalar upper) const override final
    {
        using CurveType = Bezier<Scalar, _dim, _degree, _generic>;

        if (_dim != 2) {
            throw std::runtime_error("Turning angle reduction is for 2D curves only");
        }

        std::vector<CurveType> beziers;
        std::vector<Scalar> parameter_bounds;
        std::tie(beziers, parameter_bounds) = convert_to_Bezier();
        assert(parameter_bounds.size() == beziers.size() + 1);

        std::vector<Scalar> res;
        res.reserve(static_cast<size_t>(Base::get_degree()));

        const size_t num_beziers = beziers.size();
        for (size_t idx = 0; idx < num_beziers; idx++) {
            const auto t_min = parameter_bounds[idx];
            const auto t_max = parameter_bounds[idx + 1];
            if (t_max < lower || t_min > upper) {
                continue;
            }

            Scalar normalized_lower = (lower - t_min) / (t_max - t_min);
            Scalar normalized_upper = (upper - t_min) / (t_max - t_min);
            normalized_lower = std::max<Scalar>(normalized_lower, 0);
            normalized_upper = std::min<Scalar>(normalized_upper, 1);

            const auto& curve = beziers[idx];
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
        const Scalar lower = 0.0, const Scalar upper = 1.0) const override
    {
        using CurveType = Bezier<Scalar, _dim, _degree, _generic>;

        if (_dim != 2) {
            throw std::runtime_error("Singularity computation is for 2D curves only");
        }

        std::vector<CurveType> beziers;
        std::vector<Scalar> parameter_bounds;
        std::tie(beziers, parameter_bounds) = convert_to_Bezier();
        assert(parameter_bounds.size() == beziers.size() + 1);

        std::vector<Scalar> res;
        res.reserve(static_cast<size_t>(Base::get_degree()));

        const size_t num_beziers = beziers.size();
        for (size_t idx = 0; idx < num_beziers; idx++) {
            const auto t_min = parameter_bounds[idx];
            const auto t_max = parameter_bounds[idx + 1];
            if (t_max < lower || t_min > upper) {
                continue;
            }

            Scalar normalized_lower = (lower - t_min) / (t_max - t_min);
            Scalar normalized_upper = (upper - t_min) / (t_max - t_min);
            normalized_lower = std::max<Scalar>(normalized_lower, 0);
            normalized_upper = std::min<Scalar>(normalized_upper, 1);

            const auto& curve = beziers[idx];
            auto result = curve.compute_singularities(normalized_lower, normalized_upper);
            for (auto t : result) {
                res.push_back(t * (t_max - t_min) + t_min);
            }
        }

        std::sort(res.begin(), res.end());
        res.erase(std::unique(res.begin(), res.end()), res.end());

        return res;
    }

    /**
     * Extract a subcurve in range [t0, t1].
     */
    ThisType subcurve(Scalar t0, Scalar t1) const
    {
        if (t0 > t1) {
            throw invalid_setting_error("t0 must be smaller than t1");
        }
        if (t0 < 0 || t0 > 1 || t1 < 0 || t1 > 1) {
            throw invalid_setting_error("Invalid range");
        }
        return split(t0)[1].split(t1)[0];
    }

public:
    std::tuple<std::vector<Bezier<Scalar, _dim, _degree, _generic>>, std::vector<Scalar>>
    convert_to_Bezier() const
    {
        const int d = Base::get_degree();
        const auto t_min = Base::get_domain_lower_bound();
        const auto t_max = Base::get_domain_upper_bound();

        BSpline<Scalar, _dim, _degree, _generic> curve = *this;
        {
            // Insert more knots such that all internal knots has multiplicity d.
            const auto& knots = Base::get_knots();
            const auto m = knots.size();

            {
                // Add multiplicity in end points first.
                // This ensures BSpline loops are handled correctly.
                const auto k_start = curve.locate_span(t_min);
                const auto k_end = curve.locate_span(t_max);
                const auto s_start = curve.get_multiplicity(k_start);
                const auto s_end = curve.get_multiplicity(k_end + 1);
                if (d > s_start) {
                    curve.insert_knot(t_min, d - s_start);
                }
                if (d > s_end) {
                    curve.insert_knot(t_max, d - s_end);
                }
            }

            for (int i = 0; i < m; i++) {
                if (knots[i] <= t_min) continue;
                if (knots[i] >= t_max) break;

                const int k = curve.locate_span(knots[i]);
                const int s = curve.get_multiplicity(k);
                if (d > s) {
                    curve.insert_knot(knots[i], d - s);
                }
            }
        }

        using CurveType = Bezier<Scalar, _dim, _degree, _generic>;
        std::vector<CurveType> segments;
        const auto& ctrl_pts = curve.get_control_points();
        const auto& knots = curve.get_knots();
        const auto m = knots.size();
        Scalar curr_t = t_min;
        std::vector<Scalar> parameter_bounds;
        parameter_bounds.reserve(static_cast<size_t>(m + 1));
        parameter_bounds.push_back(t_min);
        for (int i = 0; i < m; i++) {
            if (knots[i] <= curr_t) continue;
            if (knots[i] > t_max) break;
            curr_t = knots[i];

            typename CurveType::ControlPoints local_ctrl_pts(d + 1, _dim);
            local_ctrl_pts = ctrl_pts.block(i - d - 1, 0, d + 1, _dim);
            CurveType segment;
            segment.set_control_points(std::move(local_ctrl_pts));
            segments.push_back(std::move(segment));
            parameter_bounds.push_back(curr_t);
        }

        return {segments, parameter_bounds};
    }

    BSpline < _Scalar, _dim, _degree<0 ? _degree : _degree + 1, _generic> elevate_degree() const
    {
        constexpr int elevated_degree = _degree < 0 ? _degree : _degree + 1;
        using TargetType = BSpline<_Scalar, _dim, elevated_degree, _generic>;
        using BezierType = Bezier<Scalar, _dim, _degree, _generic>;
        using BezierType2 = Bezier<Scalar, _dim, elevated_degree, _generic>;

        if (Base::get_degree() == 0) {
            throw invalid_setting_error("Cannot elevate degree 0 BSpline");
        }

        std::vector<BezierType> beziers;
        std::vector<Scalar> parameter_bounds;
        std::tie(beziers, parameter_bounds) = convert_to_Bezier();

        std::vector<BezierType2> beziers2;
        beziers2.reserve(beziers.size());
        std::for_each(beziers.begin(), beziers.end(), [&beziers2](const BezierType& curve) {
            beziers2.push_back(curve.elevate_degree());
        });

        return TargetType(beziers2, parameter_bounds);
    }

private:
    template <typename Derived>
    void blossom(
        BlossomVector blossom_vector, int p, int k, Eigen::PlainObjectBase<Derived>& ctrl_pts) const
    {
        assert(ctrl_pts.rows() >= p + 1);

        for (int r = 1; r <= p; r++) {
            Scalar t = blossom_vector(r - 1);
            for (int j = p; j >= r; j--) {
                int index = j + k;
                const Scalar diff = Base::m_knots[index + 1 - r] - Base::m_knots[index - p];
                Scalar alpha = 0.0;
                if (diff > 0) {
                    alpha = (t - Base::m_knots[index - p]) / diff;
                }

                ctrl_pts.row(j) = (1.0 - alpha) * ctrl_pts.row(j - 1) + alpha * ctrl_pts.row(j);
            }
        }
    }

    template <typename Derived>
    Point deBoor(Scalar t, int p, int k, Eigen::PlainObjectBase<Derived>& ctrl_pts) const
    {
        if (p > 0) {
            // Set t_i=t for all blossom evaluation points
            BlossomVector blossom_vector(Base::get_degree());
            blossom_vector.setConstant(t);

            // Evaluate the blossom
            blossom(blossom_vector, p, k, ctrl_pts);
        }
        return ctrl_pts.row(p);
    }

    void combine_Beziers(const std::vector<Bezier<_Scalar, _dim, _degree, _generic>>& beziers,
        const std::vector<_Scalar>& parameter_bounds)
    {
        constexpr Scalar TOL = std::numeric_limits<Scalar>::epsilon();
        const int num_curves = static_cast<int>(beziers.size());
        if (num_curves == 0) {
            throw invalid_setting_error("Input must contain at least 1 Béziers curves");
        }
        assert(parameter_bounds.size() == static_cast<size_t>(num_curves + 1));
        const int degree = beziers.front().get_degree();
        const int num_ctrl_pts = num_curves * degree + 1;
        const int num_knots = num_curves * degree + degree + 2;

        Base::m_control_points.resize(num_ctrl_pts, _dim);
        Base::m_control_points.topRows(degree + 1) = beziers[0].get_control_points();
        for (int i = 1; i < num_curves; i++) {
            assert(degree == beziers[static_cast<size_t>(i)].get_degree());
            const auto& seg_ctrl_pts = beziers[static_cast<size_t>(i)].get_control_points();
            assert((seg_ctrl_pts.row(0) - Base::m_control_points.row(i * degree)).norm() < TOL);
            Base::m_control_points.block(1 + i * degree, 0, degree, _dim) =
                seg_ctrl_pts.block(1, 0, degree, _dim);
        }

        Base::m_knots.resize(num_knots);
        Base::m_knots.segment(0, degree + 1).setConstant(parameter_bounds.front());
        Base::m_knots.segment(num_knots - degree - 1, degree + 1)
            .setConstant(parameter_bounds.back());
        for (int i = 1; i < num_curves; i++) {
            Base::m_knots.segment(i * degree + 1, degree)
                .setConstant(parameter_bounds[static_cast<size_t>(i)]);
        }

        for (const auto t : parameter_bounds) {
            // Attempt to remove as many multiplicty as possible.
            Base::remove_knot(t, degree, TOL);
        }
    }
};

} // namespace nanospline
