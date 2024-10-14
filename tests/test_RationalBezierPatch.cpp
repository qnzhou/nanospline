#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include <nanospline/RationalBezierPatch.h>
#include <nanospline/forward_declaration.h>
#include <nanospline/save_msh.h>

#include "validation_utils.h"

TEST_CASE("RationalBezierPatch", "[rational][rational_bezier_patch]") {
    using namespace nanospline;
    using Scalar = double;

    SECTION("Bilinear patch") {
        RationalBezierPatch<Scalar, 3, 1, 1> patch;
        Eigen::Matrix<Scalar, 4, 3> control_grid;
        Eigen::Matrix<Scalar, 4, 1> weights;
        weights.setConstant(1);
        control_grid <<
            0.0, 0.0, 0.0,
            0.0, 1.0, 1.0,
            1.0, 0.0, 1.0,
            1.0, 1.0, 0.0;
        patch.set_control_grid(control_grid);
        patch.set_weights(weights);
        patch.initialize();

        REQUIRE(patch.get_degree_u() == 1);
        REQUIRE(patch.get_degree_v() == 1);

        const auto corner_00 = patch.evaluate(0.0, 0.0);
        const auto corner_01 = patch.evaluate(0.0, 1.0);
        const auto corner_11 = patch.evaluate(1.0, 1.0);
        const auto corner_10 = patch.evaluate(1.0, 0.0);
        REQUIRE_THAT((corner_00 - control_grid.row(0)).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
        REQUIRE_THAT((corner_01 - control_grid.row(1)).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
        REQUIRE_THAT((corner_10 - control_grid.row(2)).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
        REQUIRE_THAT((corner_11 - control_grid.row(3)).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));

        const auto p_mid = patch.evaluate(0.5, 0.5);
        REQUIRE_THAT(p_mid[0], Catch::Matchers::WithinAbs(0.5, 1e-6));
        REQUIRE_THAT(p_mid[1], Catch::Matchers::WithinAbs(0.5, 1e-6));
        REQUIRE_THAT(p_mid[2], Catch::Matchers::WithinAbs(0.5, 1e-6));

        validate_derivative(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10);
        validate_inverse_evaluation_3d(patch, 10, 10);
    }

    SECTION("Cubic patch") {
        RationalBezierPatch<Scalar, 3, 3, 3> patch;
        Eigen::Matrix<Scalar, 16, 3> control_grid;
        for (int i=0; i<4; i++) {
            for (int j=0; j<4; j++) {
                control_grid.row(i*4+j) << j, i, ((i+j)%2==0)?-1:1;
            }
        }
        patch.set_control_grid(control_grid);

        Eigen::Matrix<Scalar, 16, 1> weights;
        SECTION("Uniform weight") {
            weights.setConstant(1);
        }
        SECTION("Non-uniform weight") {
            weights.setConstant(1);
            weights[5] = 2.0;
            weights[6] = 2.0;
            weights[9] = 2.0;
            weights[10] = 2.0;
        }
        SECTION("Zero weights") {
            weights.setConstant(1.0);
            weights[2] = 1e-15;
            weights[9] = 1e-15;
        }
        patch.set_weights(weights);
        patch.initialize();

        const auto corner_00 = patch.evaluate(0.0, 0.0);
        const auto corner_01 = patch.evaluate(0.0, 1.0);
        const auto corner_11 = patch.evaluate(1.0, 1.0);
        const auto corner_10 = patch.evaluate(1.0, 0.0);
        REQUIRE_THAT((corner_00 - control_grid.row(0)).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
        REQUIRE_THAT((corner_01 - control_grid.row(3)).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
        REQUIRE_THAT((corner_10 - control_grid.row(12)).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
        REQUIRE_THAT((corner_11 - control_grid.row(15)).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));

        validate_derivative(patch, 10, 10);
        validate_iso_curves(patch, 10);
        validate_inverse_evaluation(patch, 10, 10);
        validate_inverse_evaluation_3d(patch, 10, 10);
    }

    SECTION("Extrapolation") {
        RationalBezierPatch<Scalar, 3, 3, 3> patch;
        Eigen::Matrix<Scalar, 16, 3> control_grid;
        for (int i=0; i<4; i++) {
            for (int j=0; j<4; j++) {
                control_grid.row(i*4+j) << j, i, ((i+j)%2==0)?-1:1;
            }
        }
        patch.set_control_grid(control_grid);
        Eigen::Matrix<Scalar, 16, 1> weights;
        weights.setConstant(2.0);
        patch.set_weights(weights);
        patch.initialize();

        constexpr Scalar d = 0.1;
        const auto u_min = patch.get_u_lower_bound();
        const auto u_max = patch.get_u_upper_bound();
        const auto v_min = patch.get_v_lower_bound();
        const auto v_max = patch.get_v_upper_bound();

        const auto corner_00 = patch.evaluate(u_min - d, v_min - d);
        const auto corner_01 = patch.evaluate(u_max + d, v_min - d);
        const auto corner_11 = patch.evaluate(u_max + d, v_max + d);
        const auto corner_10 = patch.evaluate(u_min - d, v_max + d);

        REQUIRE(corner_00[0] < 0);
        REQUIRE(corner_00[1] < 0);
        REQUIRE(corner_10[0] > 3);
        REQUIRE(corner_10[1] < 0);
        REQUIRE(corner_11[0] > 3);
        REQUIRE(corner_11[1] > 3);
        REQUIRE(corner_01[0] < 0);
        REQUIRE(corner_01[1] > 3);
    }

    SECTION("Periodic patch")
    {
        RationalBezierPatch<Scalar, 3, 1, 3> patch;
        Eigen::Matrix<Scalar, 8, 3> control_grid;
        control_grid << 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0;
        patch.set_control_grid(control_grid);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> weights(8);
        weights << 1.0, 0.5, 0.5, 1.0, 0.5, 1.0, 1.0, 0.5;
        patch.set_weights(weights);
        patch.set_periodic_v(true);
        patch.initialize();

        REQUIRE_THAT((patch.evaluate(0.1, 0.1) - patch.evaluate(0.1, 1.1)).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
        REQUIRE_THAT((patch.evaluate(0.3, 0.5) - patch.evaluate(0.3, 1.5)).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));

        Eigen::Matrix<Scalar, 1, 3> q(-1, -1, 0);
        auto uv0 = std::get<0>(patch.inverse_evaluate(q, 0, 1, 0, 1));
        REQUIRE_THAT(patch.evaluate(uv0[0], uv0[1]).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
        auto uv1 = std::get<0>(patch.inverse_evaluate(q, 0, 1, 1.9, 2.1));
        REQUIRE_THAT(patch.evaluate(uv1[0], uv1[1]).norm(), Catch::Matchers::WithinAbs(0.0, 1e-2));
        auto uv2 = std::get<0>(patch.inverse_evaluate(q, 0, 1, -1.1, -0.5));
        REQUIRE_THAT(patch.evaluate(uv2[0], uv2[1]).norm(), Catch::Matchers::WithinAbs(0.0, 1e-2));
        auto uv3 = std::get<0>(patch.inverse_evaluate(q, 0, 1, -1.7, -1.5));
        REQUIRE(patch.evaluate(uv3[0], uv3[1]).norm() > 0.1);
    }
}

TEST_CASE("RationalBezierPatch Benchmark", "[!benchmark][rational_bezier_patch]") {
    using namespace nanospline;
    using Scalar = double;

    RationalBezierPatch<Scalar, 3, 3, 3> patch;
    Eigen::Matrix<Scalar, 16, 3> control_grid;
    for (int i=0; i<4; i++) {
        for (int j=0; j<4; j++) {
            control_grid.row(i*4+j) << j, i, ((i+j)%2==0)?-1:1;
        }
    }
    patch.set_control_grid(control_grid);

    Eigen::Matrix<Scalar, 16, 1> weights;
    weights.setConstant(1);
    weights[5] = 2.0;
    weights[6] = 2.0;
    weights[9] = 2.0;
    weights[10] = 2.0;
    patch.set_weights(weights);
    patch.initialize();

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

