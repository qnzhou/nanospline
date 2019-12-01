#pragma once

#include <nanospline/Bezier.h>
#include <nanospline/BSpline.h>

namespace nanospline {

template<typename Scalar, int dim, int degree, bool generic>
auto compute_hodograph(const Bezier<Scalar, dim, degree, generic>& curve) {
    using HodographType = Bezier<Scalar, dim, degree<=0?degree:degree-1, generic>;
    HodographType hodograph;

    const auto& ctrl_pts = curve.get_control_points();
    const auto num_ctrl_pts = ctrl_pts.rows();
    typename HodographType::ControlPoints ctrl_pts_2(
            num_ctrl_pts-1, ctrl_pts.cols());

    if (num_ctrl_pts == 1) {
        ctrl_pts_2.resize(1, ctrl_pts.cols());
        ctrl_pts_2.setZero();
        hodograph.set_control_points(ctrl_pts_2);
        return hodograph;
    }

    for (int i=0; i<num_ctrl_pts-1; i++) {
        ctrl_pts_2.row(i) = (num_ctrl_pts-1) *
            (ctrl_pts.row(i+1) - ctrl_pts.row(i));
    }
    hodograph.set_control_points(std::move(ctrl_pts_2));
    return hodograph;
}

template<typename Scalar, int dim, int degree, bool generic>
auto compute_hodograph(const BSpline<Scalar, dim, degree, generic>& curve) {
    using HodographType = BSpline<Scalar, dim, degree<=0?degree:degree-1, generic>;
    HodographType hodograph;

    const auto d = curve.get_degree();
    const auto& knots = curve.get_knots();
    const auto& ctrl_pts = curve.get_control_points();
    const auto num_ctrl_pts = ctrl_pts.rows();

    if (d == 0) {
        typename HodographType::ControlPoints ctrl_pts_2(num_ctrl_pts, ctrl_pts.cols());
        ctrl_pts_2.setZero();
        hodograph.set_control_points(std::move(ctrl_pts_2));
        hodograph.set_knots(knots);
        return hodograph;
    }

    hodograph.set_knots(knots.segment(1, knots.rows()-2).eval());

    typename HodographType::ControlPoints ctrl_pts_2(
            num_ctrl_pts-1, ctrl_pts.cols());
    for (int i=0; i<num_ctrl_pts-1; i++) {
        const auto diff = knots[i+d+1] - knots[i+1];
        if (diff > 0) {
            ctrl_pts_2.row(i) = d * (ctrl_pts.row(i+1) - ctrl_pts.row(i)) / diff;
        } else {
            ctrl_pts_2.row(i).setZero();
        }
    }

    hodograph.set_control_points(std::move(ctrl_pts_2));
    return hodograph;
}

}
