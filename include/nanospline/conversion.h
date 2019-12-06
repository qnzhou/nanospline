#pragma once

#include <nanospline/Bezier.h>
#include <nanospline/BSpline.h>
#include <nanospline/RationalBezier.h>
#include <nanospline/NURBS.h>

namespace nanospline {

/**
 * Convert BSpline curve to a sequence of Bézier curves.
 */
template <typename Scalar, int dim, int degree, bool generic>
std::vector<Bezier<Scalar, dim, degree, generic>> convert_to_Bezier(
        BSpline<Scalar, dim, degree, generic> curve) {
    const int d = curve.get_degree();
    const auto t_min = curve.get_domain_lower_bound();
    const auto t_max = curve.get_domain_upper_bound();

    std::cout << curve.get_knots().transpose() << std::endl;
    {
        // Insert more knots such that all internal knots has multiplicity d.
        const auto knots = curve.get_knots(); // Copy on purpose
        const auto m = knots.size();

        Scalar t = t_min;
        for (int i=0; i<m; i++) {
            if (knots[i] < t_min) continue;
            if (knots[i] > t_max) break;

            const int k = curve.locate_span(knots[i]);
            const int s = curve.get_multiplicity(k);
            if (d > s) {
                curve.insert_knot(knots[i], d-s);
            }
        }
    }
    std::cout << curve.get_knots().transpose() << std::endl;

    using CurveType = Bezier<Scalar, dim, degree, generic>;
    std::vector<CurveType> segments;
    const auto& ctrl_pts = curve.get_control_points();
    const auto& knots = curve.get_knots();
    const auto m = knots.size();
    Scalar curr_t = t_min;
    for (int i=0; i<m; i++) {
        if (knots[i] <= curr_t) continue;
        if (knots[i] > t_max) break;
        curr_t = knots[i];

        typename CurveType::ControlPoints local_ctrl_pts(d+1, dim);
        local_ctrl_pts = ctrl_pts.block(i-d-1, 0, d+1, dim);
        CurveType segment;
        segment.set_control_points(std::move(local_ctrl_pts));
        segments.push_back(segment);
    }

    return segments;
}

/**
 * Convert Bézier curve to BSpline curve.
 */
template <typename Scalar, int dim, int degree, bool generic>
BSpline<Scalar, dim, degree, generic> convert_to_BSpline(
        const Bezier<Scalar, dim, degree, generic>& curve) {
    BSpline<Scalar, dim, degree, generic> out_curve;
    out_curve.set_control_points(curve.get_control_points());

    const int d = curve.get_degree();
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> knots((d + 1)*2);
    knots.segment(0, d+1).setConstant(0);
    knots.segment(d+1, d+1).setConstant(1);

    out_curve.set_knots(knots);
    return out_curve;
}

/**
 * Convert Bézier curve to rational Bézier curve.
 */
template <typename Scalar, int dim, int degree, bool generic>
RationalBezier<Scalar, dim, degree, generic> convert_to_RationalBezier(
        const Bezier<Scalar, dim, degree, generic>& curve) {
    RationalBezier<Scalar, dim, degree, generic> out_curve;
    out_curve.set_control_points(curve.get_control_points());

    const int d = curve.get_degree();
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> weights(d+1);
    weights.setConstant(1.0);
    out_curve.set_weights(weights);
    out_curve.initialize();

    return out_curve;
}


}
