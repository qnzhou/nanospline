#include <catch2/catch.hpp>
#include <iostream>

#include <nanospline/BezierPatch.h>
#include <nanospline/save_obj.h>
#include <nanospline/forward_declaration.h>

#include "validation_utils.h"

TEST_CASE("BeizerPatch", "[nonrational][bezier_patch]") {
    using namespace nanospline;
    using Scalar = double;

    SECTION("Bilinear patch") {
        BezierPatch<Scalar, 3, 1, 1> patch;
        Eigen::Matrix<Scalar, 4, 3> control_grid;
        control_grid <<
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
            1.0, 1.0, 0.0;
        patch.set_control_grid(control_grid);
        patch.initialize();

        REQUIRE(patch.get_degree_u() == 1);
        REQUIRE(patch.get_degree_v() == 1);

        const auto corner_00 = patch.evaluate(0.0, 0.0);
        const auto corner_01 = patch.evaluate(0.0, 1.0);
        const auto corner_11 = patch.evaluate(1.0, 1.0);
        const auto corner_10 = patch.evaluate(1.0, 0.0);
        REQUIRE((corner_00 - control_grid.row(0)).norm() == Approx(0.0));
        REQUIRE((corner_01 - control_grid.row(1)).norm() == Approx(0.0));
        REQUIRE((corner_10 - control_grid.row(2)).norm() == Approx(0.0));
        REQUIRE((corner_11 - control_grid.row(3)).norm() == Approx(0.0));

        const auto p_mid = patch.evaluate(0.5, 0.5);
        REQUIRE(p_mid[0] == Approx(0.5));
        REQUIRE(p_mid[1] == Approx(0.5));
        REQUIRE(p_mid[2] == Approx(0.0));

        validate_derivative(patch, 10, 10);
        validate_derivative_patches(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10);
        validate_inverse_evaluation_3d(patch, 10, 10);
    }
    SECTION("Bilinear patch non-planar") {
        BezierPatch<Scalar, 3, 1, 1> patch;
        Eigen::Matrix<Scalar, 4, 3> control_grid;
        control_grid <<
            0.0, 0.0, 0.0,
            0.0, 1.0, 1.0,
            1.0, 0.0, 1.0,
            1.0, 1.0, 0.0;
        patch.set_control_grid(control_grid);
        patch.initialize();

        REQUIRE(patch.get_degree_u() == 1);
        REQUIRE(patch.get_degree_v() == 1);

        const auto corner_00 = patch.evaluate(0.0, 0.0);
        const auto corner_01 = patch.evaluate(0.0, 1.0);
        const auto corner_11 = patch.evaluate(1.0, 1.0);
        const auto corner_10 = patch.evaluate(1.0, 0.0);
        REQUIRE((corner_00 - control_grid.row(0)).norm() == Approx(0.0));
        REQUIRE((corner_01 - control_grid.row(1)).norm() == Approx(0.0));
        REQUIRE((corner_10 - control_grid.row(2)).norm() == Approx(0.0));
        REQUIRE((corner_11 - control_grid.row(3)).norm() == Approx(0.0));

        const auto p_mid = patch.evaluate(0.5, 0.5);
        REQUIRE(p_mid[0] == Approx(0.5));
        REQUIRE(p_mid[1] == Approx(0.5));
        REQUIRE(p_mid[2] == Approx(0.5));

        //validate_iso_curves(patch, 10);
        //validate_derivative(patch, 10, 10);
        //validate_derivative_patches(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10);
        validate_inverse_evaluation_3d(patch, 10, 10);
    }
    SECTION("Cubic patch") {
        BezierPatch<Scalar, 3, 3, 3> patch;
        Eigen::Matrix<Scalar, 16, 3> control_grid;
        for (int i=0; i<4; i++) {
            for (int j=0; j<4; j++) {
                control_grid.row(i*4+j) << j, i, ((i+j)%2==0)?-1:1;
            }
        }
        patch.set_control_grid(control_grid);
        patch.initialize();

        const auto corner_00 = patch.evaluate(0.0, 0.0);
        const auto corner_01 = patch.evaluate(0.0, 1.0);
        const auto corner_11 = patch.evaluate(1.0, 1.0);
        const auto corner_10 = patch.evaluate(1.0, 0.0);
        REQUIRE((corner_00 - control_grid.row(0)).norm() == Approx(0.0));
        REQUIRE((corner_01 - control_grid.row(3)).norm() == Approx(0.0));
        REQUIRE((corner_10 - control_grid.row(12)).norm() == Approx(0.0));
        REQUIRE((corner_11 - control_grid.row(15)).norm() == Approx(0.0));

        validate_derivative(patch, 10, 10);
        validate_derivative_patches(patch, 10, 10);
        validate_iso_curves(patch, 10);
        validate_inverse_evaluation(patch, 10, 10);
        validate_inverse_evaluation_3d(patch, 10, 10);
    }
}

TEST_CASE("BezierPatch Benchmark", "[!benchmark][bezier_patch]") {
    using namespace nanospline;
    using Scalar = double;
    BezierPatch<Scalar, 3, 3, 3> patch;
    Eigen::Matrix<Scalar, 16, 3> control_grid;
    for (int i=0; i<4; i++) {
        for (int j=0; j<4; j++) {
            control_grid.row(i*4+j) << j, i, ((i+j)%2==0)?-1:1;
        }
    }
    patch.set_control_grid(control_grid);
    patch.initialize();

    SECTION("not cached") {
        INFO("Not cached");
        patch.cache_derivatives(0);
    }
    SECTION("cached") {
        INFO("Derivative and 2nd derivative cached");
        patch.cache_derivatives(2);
    }

    BENCHMARK("Evaluation") {
        return patch.evaluate(0.5, 0.6);
    };

    BENCHMARK("Derivative") {
        auto du = patch.evaluate_derivative_u(0.5, 0.6);
        auto dv = patch.evaluate_derivative_u(0.5, 0.6);
        Eigen::Matrix<Scalar, 2, 3> grad;
        grad << du, dv;
        return grad;
    };

    BENCHMARK("2nd Derivative") {
        auto duu = patch.evaluate_2nd_derivative_uu(0.5, 0.6);
        auto dvv = patch.evaluate_2nd_derivative_vv(0.5, 0.6);
        auto duv = patch.evaluate_2nd_derivative_uv(0.5, 0.6);
        Eigen::Matrix<Scalar, 3, 3> hessian;
        hessian << duu, duv, dvv;
        return hessian;
    };
}
