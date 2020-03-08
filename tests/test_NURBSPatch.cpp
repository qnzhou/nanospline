#include <catch2/catch.hpp>
#include <iostream>

#include <nanospline/NURBSPatch.h>

#include "validation_utils.h"

TEST_CASE("NURBSPatch", "[rational][bspline_patch]") {
    using namespace nanospline;
    using Scalar = double;

    SECTION("Bilinear patch non-planar") {
        NURBSPatch<Scalar, 3, 1, 1> patch;
        Eigen::Matrix<Scalar, 4, 3> control_grid;
        control_grid <<
            0.0, 0.0, 0.0,
            1.0, 0.0, 1.0,
            0.0, 1.0, 1.0,
            1.0, 1.0, 0.0;
        patch.set_control_grid(control_grid);

        Eigen::Matrix<Scalar, 4, 1> weights;
        weights.setConstant(2.0);
        patch.set_weights(weights);

        Eigen::Matrix<Scalar, 4, 1> knots_u, knots_v;
        knots_u << 0.0, 0.0, 1.0, 1.0;
        knots_v << 0.0, 0.0, 1.0, 1.0;
        patch.set_knots_u(knots_u);
        patch.set_knots_v(knots_v);

        patch.initialize();

        REQUIRE(patch.get_degree_u() == 1);
        REQUIRE(patch.get_degree_v() == 1);

        const auto corner_00 = patch.evaluate(0.0, 0.0);
        const auto corner_01 = patch.evaluate(0.0, 1.0);
        const auto corner_11 = patch.evaluate(1.0, 1.0);
        const auto corner_10 = patch.evaluate(1.0, 0.0);
        REQUIRE((corner_00 - control_grid.row(0)).norm() == Approx(0.0));
        REQUIRE((corner_10 - control_grid.row(1)).norm() == Approx(0.0));
        REQUIRE((corner_01 - control_grid.row(2)).norm() == Approx(0.0));
        REQUIRE((corner_11 - control_grid.row(3)).norm() == Approx(0.0));

        const auto p_mid = patch.evaluate(0.5, 0.5);
        REQUIRE(p_mid[0] == Approx(0.5));
        REQUIRE(p_mid[1] == Approx(0.5));
        REQUIRE(p_mid[2] == Approx(0.5));

        validate_derivative(patch, 10, 10);
    }

    SECTION("Cubic patch") {
        NURBSPatch<Scalar, 3, 3, 3> patch;
        Eigen::Matrix<Scalar, 16, 3> control_grid;
        for (int i=0; i<4; i++) {
            for (int j=0; j<4; j++) {
                control_grid.row(i*4+j) << j, i, ((i+j)%2==0)?-1:1;
            }
        }
        patch.set_control_grid(control_grid);
        Eigen::Matrix<Scalar, 8, 1> knots_u, knots_v;
        knots_u << 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0;
        knots_v << 0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 2.5;
        patch.set_knots_u(knots_u);
        patch.set_knots_v(knots_v);

        Eigen::Matrix<Scalar, 16, 1> weights;
        SECTION("Uniform weight") {
            weights.setConstant(1.0);
        }
        SECTION("Non-uniform weight") {
            weights[5] = 2.0;
            weights[6] = 2.0;
            weights[9] = 2.0;
            weights[10] = 2.0;
        }
        patch.set_weights(weights);
        patch.initialize();

        const auto u_min = patch.get_u_lower_bound();
        const auto u_max = patch.get_u_upper_bound();
        const auto v_min = patch.get_v_lower_bound();
        const auto v_max = patch.get_v_upper_bound();

        SECTION("Isocurves") {
            for (int i=0; i<=10; i++) {
                for (int j=0; j<=10; j++) {
                    const Scalar u = 0.1 * i * (u_max-u_min) + u_min;
                    const Scalar v = 0.1 * j * (v_max-v_min) + v_min;
                    auto u_curve = patch.compute_iso_curve_u(v);
                    auto v_curve = patch.compute_iso_curve_v(u);
                    auto p = patch.evaluate(u, v);
                    auto p1 = u_curve.evaluate(u);
                    auto p2 = v_curve.evaluate(v);
                    REQUIRE((p-p1).norm() == Approx(0.0).margin(1e-6));
                    REQUIRE((p-p2).norm() == Approx(0.0).margin(1e-6));
                }
            }
        }

        validate_derivative(patch, 10, 10);
    }

}

