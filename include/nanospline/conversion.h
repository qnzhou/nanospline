#pragma once

#include <nanospline/Bezier.h>
#include <nanospline/BSpline.h>
#include <nanospline/RationalBezier.h>
#include <nanospline/NURBS.h>
#include <nanospline/Exceptions.h>

namespace nanospline {

/**
 * Convert BSpline curve to a sequence of Bézier curves.
 */
template <typename Scalar, int dim, int degree, bool generic>
std::vector<Bezier<Scalar, dim, degree, generic>> convert_to_Bezier(
        BSpline<Scalar, dim, degree, generic>& curve) {
    return std::get<0>(curve.convert_to_Bezier());
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

/**
 * Convert Rational Bézier curve to Bézier curve if possible.
 */
template <typename Scalar, int dim, int degree, bool generic>
Bezier<Scalar, dim, degree, generic> convert_to_Bezier(
        const RationalBezier<Scalar, dim, degree, generic>& curve) {
    const auto& weights = curve.get_weights();
    if (weights.size() > 0 && (weights.array() != weights[0]).any()) {
        throw invalid_setting_error("Invalid conversion!");
    }

    Bezier<Scalar, dim, degree, generic> out_curve;
    out_curve.set_control_points(curve.get_control_points());
    return out_curve;
}

/**
 * Convert BSpline curve to NURBS curve.
 */
template <typename Scalar, int dim, int degree, bool generic>
NURBS<Scalar, dim, degree, generic> convert_to_NURBS(
        const BSpline<Scalar, dim, degree, generic>& curve) {
    NURBS<Scalar, dim, degree, generic> out_curve;
    const auto& control_points = curve.get_control_points();
    out_curve.set_control_points(control_points);
    out_curve.set_knots(curve.get_knots());

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> weights(control_points.rows());
    weights.setConstant(1.0);
    out_curve.set_weights(weights);
    out_curve.initialize();

    return out_curve;
}

/**
 * Convert NURBS curve to BSpline curve.
 */
template <typename Scalar, int dim, int degree, bool generic>
BSpline<Scalar, dim, degree, generic> convert_to_BSpline(
        const NURBS<Scalar, dim, degree, generic>& curve) {
    const auto& weights = curve.get_weights();
    if (weights.size() > 0 && (weights.array() != weights[0]).any()) {
        throw invalid_setting_error("Invalid conversion!");
    }

    BSpline<Scalar, dim, degree, generic> out_curve;
    out_curve.set_control_points(curve.get_control_points());
    out_curve.set_knots(curve.get_knots());
    return out_curve;
}

}
