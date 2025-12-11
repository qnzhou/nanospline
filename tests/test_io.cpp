#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <string>
#include <vector>

#include "validation_utils.h"

#include <nanospline/save_msh.h>
#include <nanospline/load_msh.h>
#include <nanospline/Bezier.h>
#include <nanospline/BSpline.h>
#include <nanospline/RationalBezier.h>
#include <nanospline/NURBS.h>
#include <nanospline/BezierPatch.h>
#include <nanospline/BSplinePatch.h>
#include <nanospline/RationalBezierPatch.h>
#include <nanospline/NURBSPatch.h>

#if defined(NANOSPLINE_MSHIO)

TEST_CASE("IO", "[io][msh]")
{
    using namespace nanospline;
    using Scalar = double;

    Bezier<Scalar, 3, 3, true> curve1;
    {
        Eigen::Matrix<Scalar, 4, 3> control_pts;
        control_pts << 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 0.0, 3.0, 0.0, 0.0;
        curve1.set_control_points(control_pts);
    }

    NURBS<Scalar, 3, 3, true> curve2;
    {
        Eigen::Matrix<Scalar, 4, 3> control_pts;
        control_pts << 0.0, 3.0, 0.0, 1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 3.0, 0.0, 0.0;
        Eigen::Matrix<Scalar, 8, 1> knots;
        knots << 0, 0, 0, 0, 2, 2, 2, 2;

        Eigen::Matrix<Scalar, 4, 1> weights;
        weights.setOnes();

        curve2.set_control_points(control_pts);
        curve2.set_weights(weights);
        curve2.set_knots(knots);
        curve2.initialize();
    }

    NURBSPatch<Scalar, 3, -1, -1> patch1;
    {
        Eigen::Matrix<Scalar, 4, 3> control_grid;
        control_grid << 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0;
        patch1.set_control_grid(control_grid);

        Eigen::Matrix<Scalar, 4, 1> weights;
        weights.setConstant(2.0);
        patch1.set_weights(weights);

        Eigen::Matrix<Scalar, 4, 1> knots_u, knots_v;
        knots_u << 0.0, 0.0, 1.0, 1.0;
        knots_v << 0.0, 0.0, 1.0, 1.0;
        patch1.set_knots_u(knots_u);
        patch1.set_knots_v(knots_v);

        patch1.set_degree_u(1);
        patch1.set_degree_v(1);

        patch1.initialize();
    }

    BezierPatch<Scalar, 3, -1, -1> patch2;
    {
        Eigen::Matrix<Scalar, 4, 3> control_grid;
        control_grid <<
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
            1.0, 1.0, 0.0;
        patch2.set_control_grid(control_grid);
        patch2.set_degree_u(1);
        patch2.set_degree_v(1);
        patch2.initialize();
    }

    save_msh<Scalar, 3>("tmp.msh", {&curve1, &curve2}, {&patch1, &patch2}, true);

    auto r = load_msh<Scalar, 3>("tmp.msh");
    auto& out_curves = std::get<0>(r);
    auto& out_patches = std::get<1>(r);

    REQUIRE(out_curves.size() == 2);
    REQUIRE(out_patches.size() == 2);

    assert_same(curve1, *out_curves[0], 10);
    assert_same(curve2, *out_curves[1], 10);
    assert_same(patch1, *out_patches[0], 10,
            patch1.get_u_lower_bound(),
            patch1.get_u_upper_bound(),
            patch1.get_v_lower_bound(),
            patch1.get_v_upper_bound(),
            patch1.get_u_lower_bound(),
            patch1.get_u_upper_bound(),
            patch1.get_v_lower_bound(),
            patch1.get_v_upper_bound());
    assert_same(patch2, *out_patches[1], 10,
            patch2.get_u_lower_bound(),
            patch2.get_u_upper_bound(),
            patch2.get_v_lower_bound(),
            patch2.get_v_upper_bound(),
            patch2.get_u_lower_bound(),
            patch2.get_u_upper_bound(),
            patch2.get_v_lower_bound(),
            patch2.get_v_upper_bound());
}
#endif // defined(NANOSPLINE_MSHIO)
