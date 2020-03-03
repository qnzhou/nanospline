#pragma once

#include <vector>

#include <Eigen/Core>

#include <nanospline/Exceptions.h>
#include <nanospline/BSplineBase.h>
#include <nanospline/Bezier.h>

namespace nanospline {

template<typename _Scalar, int _dim=3, int _degree=3, bool _generic=_degree<0 >
class BSpline : public BSplineBase<_Scalar, _dim, _degree, _generic> {
    public:
        using Base = BSplineBase<_Scalar, _dim, _degree, _generic>;
        using Scalar = typename Base::Scalar;
        using Point = typename Base::Point;
        using ControlPoints = typename Base::ControlPoints;
        using KnotVector = typename Base::KnotVector;

    public:
        Point evaluate(Scalar t) const override {
            assert(Base::in_domain(t));
            Base::validate_curve();
            const int p = Base::get_degree();
            const int k = Base::locate_span(t);
            assert(p >= 0);
            assert(Base::m_knots.rows() ==
                    Base::m_control_points.rows() + p + 1);
            assert(Base::m_knots[k] <= t);
            assert(Base::m_knots[k+1] >= t);

            ControlPoints ctrl_pts(p+1, _dim);
            for (int i=0; i<=p; i++) {
                ctrl_pts.row(i) = Base::m_control_points.row(i+k-p);
            }

            deBoor(t, p, k, ctrl_pts);
            return ctrl_pts.row(p);
        }

        Scalar inverse_evaluate(const Point& p) const override {
            throw not_implemented_error("Too complex, sigh");
        }

        Point evaluate_derivative(Scalar t) const override {
            assert(Base::in_domain(t));
            Base::validate_curve();
            const int p = Base::get_degree();
            const int k = Base::locate_span(t);
            assert(p >= 0);
            assert(Base::m_knots.rows() ==
                    Base::m_control_points.rows() + p + 1);
            assert(Base::m_knots[k] <= t);
            assert(Base::m_knots[k+1] >= t);

            if (p == 0) return Point::Zero();

            ControlPoints ctrl_pts(p, _dim);
            for(int i=0; i<p; i++) {
                const Scalar diff = Base::m_knots[i+k+1] - Base::m_knots[i+k-p+1];
                Scalar alpha = 0.0;
                if (diff > 0) {
                    alpha = p / diff;
                }
                ctrl_pts.row(i) = alpha * (
                        Base::m_control_points.row(i+k-p+1) -
                        Base::m_control_points.row(i+k-p));
            }

            deBoor(t, p-1, k, ctrl_pts);
            return ctrl_pts.row(p-1);
        }

        Point evaluate_2nd_derivative(Scalar t) const override {
            assert(Base::in_domain(t));
            Base::validate_curve();
            const int p = Base::get_degree();
            const int k = Base::locate_span(t);
            assert(p >= 0);
            assert(Base::m_knots.rows() ==
                    Base::m_control_points.rows() + p + 1);
            assert(Base::m_knots[k] <= t);
            assert(Base::m_knots[k+1] >= t);

            if (p <= 1) return Point::Zero();

            ControlPoints ctrl_pts(p, _dim);

            // First derivative control pts.
            for(int i=0; i<p; i++) {
                const Scalar diff = Base::m_knots[i+k+1] - Base::m_knots[i+k-p+1];
                Scalar alpha = 0.0;
                if (diff > 0) {
                    alpha = p / diff;
                }
                ctrl_pts.row(i) = alpha * (
                        Base::m_control_points.row(i+k-p+1) -
                        Base::m_control_points.row(i+k-p));
            }

            // Second derivative control pts.
            for (int i=0; i<p-1; i++) {
                const Scalar diff = Base::m_knots[i+k+1] - Base::m_knots[i+k-p+2];
                Scalar alpha = 0.0;
                if (diff > 0) {
                    alpha = (p-1) / diff;
                }
                ctrl_pts.row(i) = alpha * (ctrl_pts.row(i+1) - ctrl_pts.row(i));
            }

            deBoor(t, p-2, k, ctrl_pts);
            return ctrl_pts.row(p-2);
        }

        std::tuple<std::vector<Bezier<Scalar, _dim, _degree, _generic>>, std::vector<Scalar>>
        convert_to_Bezier() const {
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
                    const auto s_end = curve.get_multiplicity(k_end+1);
                    if (d > s_start) {
                        curve.insert_knot(t_min, d-s_start);
                    }
                    if (d > s_end) {
                        curve.insert_knot(t_max, d-s_end);
                    }
                }

                for (int i=0; i<m; i++) {
                    if (knots[i] <= t_min) continue;
                    if (knots[i] >= t_max) break;

                    const int k = curve.locate_span(knots[i]);
                    const int s = curve.get_multiplicity(k);
                    if (d > s) {
                        curve.insert_knot(knots[i], d-s);
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
            parameter_bounds.reserve(static_cast<size_t>(m+1));
            parameter_bounds.push_back(t_min);
            for (int i=0; i<m; i++) {
                if (knots[i] <= curr_t) continue;
                if (knots[i] > t_max) break;
                curr_t = knots[i];

                typename CurveType::ControlPoints local_ctrl_pts(d+1, _dim);
                local_ctrl_pts = ctrl_pts.block(i-d-1, 0, d+1, _dim);
                CurveType segment;
                segment.set_control_points(std::move(local_ctrl_pts));
                segments.push_back(std::move(segment));
                parameter_bounds.push_back(curr_t);
            }

            return {segments, parameter_bounds};
        }

        std::vector<Scalar> compute_inflections (
                const Scalar lower,
                const Scalar upper) const override {
            using CurveType = Bezier<Scalar, _dim, _degree, _generic>;
            std::vector<CurveType> beziers;
            std::vector<Scalar> parameter_bounds;
            std::tie(beziers, parameter_bounds) = convert_to_Bezier();
            assert(parameter_bounds.size() == beziers.size() + 1);

            std::vector<Scalar> res;
            res.reserve(static_cast<size_t>(Base::get_degree()));

            const size_t num_beziers = beziers.size();
            for (size_t idx=0; idx < num_beziers; idx++) {
                const auto t_min = parameter_bounds[idx];
                const auto t_max = parameter_bounds[idx+1];
                if (t_max < lower || t_min > upper) {
                    continue;
                }

                Scalar normalized_lower = (lower - t_min) / (t_max - t_min);
                Scalar normalized_upper = (upper - t_min) / (t_max - t_min);
                normalized_lower = std::max<Scalar>(normalized_lower, 0);
                normalized_upper = std::min<Scalar>(normalized_upper, 1);

                const auto& curve = beziers[idx];
                auto inflections = curve.compute_inflections(
                        normalized_lower, normalized_upper);
                for (auto t : inflections) {
                    res.push_back(t * (t_max-t_min) + t_min);
                }
            }

            std::sort(res.begin(), res.end());
            res.erase(std::unique(res.begin(), res.end()), res.end());

            return res;
        }

    private:
        template<typename Derived>
        void deBoor(Scalar t, int p, int k,
                Eigen::PlainObjectBase<Derived>& ctrl_pts) const {
            assert(ctrl_pts.rows() >= p+1);

            for (int r=1; r<=p; r++) {
                for (int j=p; j>=r; j--) {
                    const Scalar diff =
                        Base::m_knots[j+1+k-r] - Base::m_knots[j+k-p];
                    Scalar alpha = 0.0;
                    if (diff > 0) {
                        alpha = (t - Base::m_knots[j+k-p]) / diff;
                    }

                    ctrl_pts.row(j) = (1.0-alpha) * ctrl_pts.row(j-1) +
                        alpha * ctrl_pts.row(j);
                }
            }
        }
};

}
