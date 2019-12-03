#pragma once

#include <cassert>
#include <vector>
#include <Eigen/Eigenvalues>

#include <nanospline/Exceptions.h>
#include <nanospline/Bezier.h>

namespace nanospline {

/**
 * Compute inflection points for 2D cubic Bézier curves.
 */
template<typename Scalar, int _degree=3, bool generic=_degree<0 >
std::vector<Scalar> compute_inflections(
        const Bezier<Scalar, 2, _degree, generic>& curve,
        Scalar t0=0.0,
        Scalar t1=1.0) {

    if (curve.get_degree() != 3) {
        throw not_implemented_error(
                "Inflection computation only works on cubic Bézier curve");
    }

    constexpr Scalar tol = 1e-8;
    const auto& ctrl_pts = curve.get_control_points();
    const Scalar cx0 = ctrl_pts(0, 0);
    const Scalar cy0 = ctrl_pts(0, 1);
    const Scalar cx1 = ctrl_pts(1, 0);
    const Scalar cy1 = ctrl_pts(1, 1);
    const Scalar cx2 = ctrl_pts(2, 0);
    const Scalar cy2 = ctrl_pts(2, 1);
    const Scalar cx3 = ctrl_pts(3, 0);
    const Scalar cy3 = ctrl_pts(3, 1);

	Eigen::Matrix<Scalar, 2, 2> companion;
	companion.setZero();
    companion(1, 0) = 1;

    companion(0, 1) = (cx0*cy1 - cx0*cy2 - cx1*cy0 + cx1*cy2 + cx2*cy0 - cx2*cy1)/(cx0*cy1 - 2*cx0*cy2 + cx0*cy3 - cx1*cy0 + 3*cx1*cy2 - 2*cx1*cy3 + 2*cx2*cy0 - 3*cx2*cy1 + cx2*cy3 - cx3*cy0 + 2*cx3*cy1 - cx3*cy2);
	companion(1, 1) = (-2*cx0*cy1 + 3*cx0*cy2 - cx0*cy3 + 2*cx1*cy0 - 3*cx1*cy2 + cx1*cy3 - 3*cx2*cy0 + 3*cx2*cy1 + cx3*cy0 - cx3*cy1)/(cx0*cy1 - 2*cx0*cy2 + cx0*cy3 - cx1*cy0 + 3*cx1*cy2 - 2*cx1*cy3 + 2*cx2*cy0 - 3*cx2*cy1 + cx2*cy3 - cx3*cy0 + 2*cx3*cy1 - cx3*cy2);

    assert(std::isfinite(companion(0, 1)));
    assert(std::isfinite(companion(1, 1)));

	Eigen::EigenSolver<Eigen::Matrix<Scalar, 2, 2>> es(companion, false);
	const auto &vals = es.eigenvalues();

    std::vector<Scalar> inflections;
    inflections.reserve(vals.size());
	for(int i = 0; i < vals.size(); ++i){
		const auto lambda = vals(i);
		const Scalar current_t = lambda.real();
		if( abs(abs(lambda)-abs(current_t)) > tol) continue;
		if(current_t >= t0 && current_t <= t1)
			inflections.push_back(current_t);
	}

    return inflections;
}

}
