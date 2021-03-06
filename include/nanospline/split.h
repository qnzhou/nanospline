#pragma once

#include <vector>

#include <nanospline/BSpline.h>
#include <nanospline/Bezier.h>
#include <nanospline/Exceptions.h>
#include <nanospline/NURBS.h>
#include <nanospline/RationalBezier.h>
#include "nanospline/PatchBase.h"

namespace nanospline {

template <typename Scalar, int dim, int degree, bool generic>
std::vector<Bezier<Scalar, dim, degree, generic>> split(
    const Bezier<Scalar, dim, degree, generic>& curve, Scalar t)
{
    using CurveType = Bezier<Scalar, dim, degree, generic>;
    if (!curve.in_domain(t)) {
        throw invalid_setting_error("Parameter not inside of the domain.");
    }
    if (t == curve.get_domain_lower_bound()) {
        return std::vector<CurveType>{curve};
    }
    if (t == curve.get_domain_upper_bound()) {
        return std::vector<CurveType>{curve};
    }
    auto r = curve.split(t);
    return {r[0], r[1]};
}

template <typename Scalar, int dim, int degree, bool generic>
std::vector<RationalBezier<Scalar, dim, degree, generic>> split(
    const RationalBezier<Scalar, dim, degree, generic>& curve, Scalar t)
{
    using CurveType = RationalBezier<Scalar, dim, degree, generic>;
    const auto homogeneous = curve.get_homogeneous();
    const auto parts = split(homogeneous, t);
    std::vector<CurveType> results;
    results.reserve(2);
    for (const auto& c : parts) {
        results.emplace_back();
        results.back().set_homogeneous(c);
    }
    return results;
}


template <typename Scalar, int dim, int degree, bool generic>
std::vector<BSpline<Scalar, dim, degree, generic>> split(
    BSpline<Scalar, dim, degree, generic> curve, Scalar t)
{
    using CurveType = BSpline<Scalar, dim, degree, generic>;
    if (!curve.in_domain(t)) {
        throw invalid_setting_error("Parameter not inside of the domain.");
    }
    if (t == curve.get_domain_lower_bound()) {
        return std::vector<CurveType>{curve};
    }
    if (t == curve.get_domain_upper_bound()) {
        return std::vector<CurveType>{curve};
    }

    const auto d = curve.get_degree();
    {
        const auto& knots = curve.get_knots();
        const int k = curve.locate_span(t);
        const int s = (t == knots[k]) ? curve.get_multiplicity(k) : 0;

        if (d > s) {
            curve.insert_knot(t, d - s);
        }
    }

    const auto& ctrl_pts = curve.get_control_points();
    const int n = static_cast<int>(ctrl_pts.rows() - 1);
    const auto& knots = curve.get_knots();
    const int m = static_cast<int>(knots.rows() - 1);
    const int k = curve.locate_span(t);

    typename CurveType::ControlPoints ctrl_pts_1(k - d + 1, dim);
    ctrl_pts_1 = ctrl_pts.topRows(k - d + 1);

    typename CurveType::ControlPoints ctrl_pts_2(n - k + d + 1, dim);
    ctrl_pts_2 = ctrl_pts.bottomRows(n - k + d + 1);

    typename CurveType::KnotVector knots_1(k + 2, 1);
    knots_1.segment(0, k + 1) = knots.segment(0, k + 1);
    knots_1[k + 1] = knots_1[k];

    typename CurveType::KnotVector knots_2(m - k + d + 1);
    knots_2.segment(1, m - k + d) = knots.segment(k - d + 1, m - k + d);
    knots_2[0] = knots_2[1];

    std::vector<CurveType> results(2);
    results[0].set_control_points(std::move(ctrl_pts_1));
    results[1].set_control_points(std::move(ctrl_pts_2));
    results[0].set_knots(std::move(knots_1));
    results[1].set_knots(std::move(knots_2));

    return results;
}

template <typename Scalar, int dim, int degree, bool generic>
std::vector<NURBS<Scalar, dim, degree, generic>> split(
    NURBS<Scalar, dim, degree, generic> curve, Scalar t)
{
    using CurveType = NURBS<Scalar, dim, degree, generic>;
    const auto& homogeneous = curve.get_homogeneous();
    const auto parts = split(homogeneous, t);
    std::vector<CurveType> results;
    results.reserve(2);
    for (const auto& c : parts) {
        results.emplace_back();
        results.back().set_homogeneous(c);
    }
    return results;
}
} // namespace nanospline
