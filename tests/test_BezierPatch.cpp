#include <catch2/catch.hpp>
#include <iostream>

#include <nanospline/BezierPatch.h>
#include <nanospline/forward_declaration.h>
#include <nanospline/save_msh.h>

#include "validation_utils.h"

TEST_CASE("BezierPatch", "[nonrational][bezier_patch]")
{
    using namespace nanospline;
    using Scalar = double;

    SECTION("Bilinear patch")
    {
        BezierPatch<Scalar, 3, 1, 1> patch;
        Eigen::Matrix<Scalar, 4, 3> control_grid;
        control_grid << 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0;
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
        offset_and_validate(patch);
    }

    SECTION("Bilinear patch non-planar")
    {
        BezierPatch<Scalar, 3, 1, 1> patch;
        Eigen::Matrix<Scalar, 4, 3> control_grid;
        control_grid << 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0;
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

        validate_iso_curves(patch, 10);
        validate_derivative(patch, 10, 10);
        validate_derivative_patches(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10);
        validate_inverse_evaluation_3d(patch, 10, 10);
        offset_and_validate(patch);
    }
    SECTION("Cubic patch")
    {
        BezierPatch<Scalar, 3, 3, 3> patch;
        Eigen::Matrix<Scalar, 16, 3> control_grid;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                control_grid.row(i * 4 + j) << j, i, ((i + j) % 2 == 0) ? -1 : 1;
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

    SECTION("Extrapolation")
    {
        BezierPatch<Scalar, 3, 3, 3> patch;
        Eigen::Matrix<Scalar, 16, 3> control_grid;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                control_grid.row(i * 4 + j) << j, i, ((i + j) % 2 == 0) ? -1 : 1;
            }
        }
        patch.set_control_grid(control_grid);
        patch.initialize();
        validate_derivative(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10);
        validate_inverse_evaluation_3d(patch, 10, 10);

        constexpr Scalar d = 0.1;
        const auto u_min = patch.get_u_lower_bound();
        const auto u_max = patch.get_u_upper_bound();
        const auto v_min = patch.get_v_lower_bound();
        const auto v_max = patch.get_v_upper_bound();

        const auto corner_00 = patch.evaluate(u_min - d, v_min - d);
        const auto corner_10 = patch.evaluate(u_max + d, v_min - d);
        const auto corner_11 = patch.evaluate(u_max + d, v_max + d);
        const auto corner_01 = patch.evaluate(u_min - d, v_max + d);

        REQUIRE(corner_00[0] < 0);
        REQUIRE(corner_00[1] < 0);
        REQUIRE(corner_01[0] > 3);
        REQUIRE(corner_01[1] < 0);
        REQUIRE(corner_11[0] > 3);
        REQUIRE(corner_11[1] > 3);
        REQUIRE(corner_10[0] < 0);
        REQUIRE(corner_10[1] > 3);

        auto uv_00 = std::get<0>(
            patch.inverse_evaluate(corner_00, u_min - d, u_max + d, v_min - d, v_max + d));
        auto uv_01 = std::get<0>(
            patch.inverse_evaluate(corner_01, u_min - d, u_max + d, v_min - d, v_max + d));
        auto uv_10 = std::get<0>(
            patch.inverse_evaluate(corner_10, u_min - d, u_max + d, v_min - d, v_max + d));
        auto uv_11 = std::get<0>(
            patch.inverse_evaluate(corner_11, u_min - d, u_max + d, v_min - d, v_max + d));

        REQUIRE(uv_00[0] == Approx(0));
        REQUIRE(uv_00[1] == Approx(0));
        REQUIRE(uv_01[0] == Approx(0));
        REQUIRE(uv_01[1] == Approx(1));
        REQUIRE(uv_10[0] == Approx(1));
        REQUIRE(uv_10[1] == Approx(0));
        REQUIRE(uv_11[1] == Approx(1));
        REQUIRE(uv_11[1] == Approx(1));
    }

    SECTION("Periodic patch")
    {
        BezierPatch<Scalar, 3, 1, 3> patch;
        Eigen::Matrix<Scalar, 8, 3> control_grid;
        control_grid << 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0;
        patch.set_control_grid(control_grid);
        patch.set_periodic_v(true);
        patch.initialize();

        REQUIRE(
            (patch.evaluate(0.1, 0.1) - patch.evaluate(0.1, 1.1)).norm() == Approx(0).margin(1e-6));
        REQUIRE(
            (patch.evaluate(0.3, 0.5) - patch.evaluate(0.3, 1.5)).norm() == Approx(0).margin(1e-6));

        Eigen::Matrix<Scalar, 1, 3> q(-1, -1, 0);
        auto uv0 = std::get<0>(patch.inverse_evaluate(q, 0, 1, 0, 1));
        REQUIRE(patch.evaluate(uv0[0], uv0[1]).norm() == Approx(0));
        auto uv1 = std::get<0>(patch.inverse_evaluate(q, 0, 1, 1.9, 2.1));
        REQUIRE(patch.evaluate(uv1[0], uv1[1]).norm() == Approx(0).margin(1e-2));
        auto uv2 = std::get<0>(patch.inverse_evaluate(q, 0, 1, -1.1, -0.5));
        REQUIRE(patch.evaluate(uv2[0], uv2[1]).norm() == Approx(0).margin(1e-2));
        auto uv3 = std::get<0>(patch.inverse_evaluate(q, 0, 1, -1.7, -1.5));
        REQUIRE(patch.evaluate(uv3[0], uv3[1]).norm() > 0.1);
    }
}

TEST_CASE("BezierPatch Benchmark", "[!benchmark][bezier_patch]")
{
    using namespace nanospline;
    using Scalar = double;
    BezierPatch<Scalar, 3, 3, 3> patch;
    Eigen::Matrix<Scalar, 16, 3> control_grid;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            control_grid.row(i * 4 + j) << j, i, ((i + j) % 2 == 0) ? -1 : 1;
        }
    }
    patch.set_control_grid(control_grid);
    patch.initialize();

    BENCHMARK("Evaluation") { return patch.evaluate(0.5, 0.6); };

    BENCHMARK("Derivative")
    {
        auto du = patch.evaluate_derivative_u(0.5, 0.6);
        auto dv = patch.evaluate_derivative_u(0.5, 0.6);
        Eigen::Matrix<Scalar, 2, 3> grad;
        grad << du, dv;
        return grad;
    };

    BENCHMARK("2nd Derivative")
    {
        auto duu = patch.evaluate_2nd_derivative_uu(0.5, 0.6);
        auto dvv = patch.evaluate_2nd_derivative_vv(0.5, 0.6);
        auto duv = patch.evaluate_2nd_derivative_uv(0.5, 0.6);
        Eigen::Matrix<Scalar, 3, 3> hessian;
        hessian << duu, duv, dvv;
        return hessian;
    };
}
