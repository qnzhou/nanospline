#include <catch2/catch.hpp>
#include <iostream>

#include <nanospline/Line.h>
#include <nanospline/NURBSPatch.h>
#include <nanospline/forward_declaration.h>
#include <nanospline/load_msh.h>
#include <nanospline/sample.h>
#include <nanospline/save_msh.h>
#include <nanospline/save_obj.h>

#include "validation_utils.h"

TEST_CASE("NURBSPatch", "[rational][nurbs_patch]")
{
    using namespace nanospline;
    using Scalar = double;
    SECTION("Bilinear patch non-planar")
    {
        NURBSPatch<Scalar, 3, 1, 1> patch;
        Eigen::Matrix<Scalar, 4, 3> control_grid;
        control_grid << 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0;
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
        REQUIRE((corner_01 - control_grid.row(1)).norm() == Approx(0.0));
        REQUIRE((corner_10 - control_grid.row(2)).norm() == Approx(0.0));
        REQUIRE((corner_11 - control_grid.row(3)).norm() == Approx(0.0));

        const auto p_mid = patch.evaluate(0.5, 0.5);
        REQUIRE(p_mid[0] == Approx(0.5));
        REQUIRE(p_mid[1] == Approx(0.5));
        REQUIRE(p_mid[2] == Approx(0.5));

        validate_derivative(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10);
        validate_inverse_evaluation_3d(patch, 10, 10);
    }

    SECTION("Periodic patch")
    {
        NURBSPatch<Scalar, 3, 2, 1> patch;
        Eigen::Matrix<Scalar, 14, 3> control_pts;
        control_pts << 0.0, 3.907985046680551e-14, 42.05086827278133, 0.0, 22.86057786410438,
            6.756824834631177, 0.0, 3.907985046680551e-14, 42.05086827278133, 20.64450354440619,
            22.86057786410438, 6.756824834631177, 0.0, 3.907985046680551e-14, 42.05086827278133,
            10.3222517722031, 5.715144466026113, 1.6892062086577977, 0.0, 3.907985046680551e-14,
            42.05086827278133, 2.9193399033287708e-15, -11.430288932052154, -3.378412417315582, 0.0,
            3.907985046680551e-14, 42.05086827278133, -10.322251772203094, 5.715144466026106,
            1.6892062086577952, 0.0, 3.907985046680551e-14, 42.05086827278133, -20.644503544406202,
            22.86057786410436, 6.756824834631171, 0.0, 3.907985046680551e-14, 42.05086827278133,
            -2.9193399033287716e-15, 22.86057786410438, 6.756824834631177;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> u_knots(10);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> v_knots(4);
        u_knots << 0.0, 0.0, 0.0, 2.0943951023931953, 2.0943951023931953, 4.1887902047863905,
            4.1887902047863905, 6.283185307179586, 6.283185307179586, 6.283185307179586;
        v_knots << -42.05086827278134, -42.05086827278134, 0.0, 0.0;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> weights(14);
        weights << 1.0, 1.0, 0.5000000000000001, 0.5000000000000001, 1.0, 1.0, 0.5000000000000001,
            0.5000000000000001, 1.0, 1.0, 0.5000000000000001, 0.5000000000000001, 1.0, 1.0;
        patch.set_control_grid(control_pts);
        patch.set_knots_u(u_knots);
        patch.set_knots_v(v_knots);
        patch.set_weights(weights);
        patch.set_periodic_u(true);
        patch.initialize();
        validate_derivative(patch, 10, 10, 1e-3);

        const auto u_min = patch.get_u_lower_bound();
        const auto u_max = patch.get_u_upper_bound();
        const auto v_min = patch.get_v_lower_bound();
        const auto v_max = patch.get_v_upper_bound();

        const auto v_mid = (v_max + v_min) * 0.5;
        const auto period = u_max - u_min;
        const auto d = period / 5;

        REQUIRE((patch.evaluate(u_min + d, v_mid) - patch.evaluate(u_min + 10 * period + d, v_mid))
                    .norm() == Approx(0).margin(1e-6));
        REQUIRE((patch.evaluate(u_min - 2 * period + d, v_mid) -
                    patch.evaluate(u_min + period + d, v_mid))
                    .norm() == Approx(0).margin(1e-6));

        auto q = patch.evaluate(u_min, v_mid);
        constexpr Scalar TOL = std::numeric_limits<Scalar>::epsilon() * 100;
        const auto uv0 = std::get<0>(patch.inverse_evaluate(q, u_min, u_max, v_min, v_max));
        REQUIRE((patch.evaluate(uv0[0], uv0[1]) - q).norm() == Approx(0).margin(TOL));
        const auto uv1 = std::get<0>(patch.inverse_evaluate(q, u_max - d, u_max + d, v_min, v_max));
        REQUIRE((patch.evaluate(uv1[0], uv1[1]) - q).norm() == Approx(0).margin(TOL));
        const auto uv2 = std::get<0>(patch.inverse_evaluate(
            q, u_min - 2 * period - d, u_min - 2 * period + d, v_min, v_max));
        REQUIRE((patch.evaluate(uv2[0], uv2[1]) - q).norm() == Approx(0).margin(TOL));
        const auto uv3 = std::get<0>(
            patch.inverse_evaluate(q, u_min - period + d, u_max - period - d, v_min, v_max));
        REQUIRE((patch.evaluate(uv3[0], uv3[1]) - q).norm() > 0.1);
    }
}

TEST_CASE("NURBSPatch 2", "[rational][nurbs_patch]")
{
    using namespace nanospline;
    using Scalar = double;

    SECTION("Cubic patch")
    {
        NURBSPatch<Scalar, 3, 3, 3> patch;
        Eigen::Matrix<Scalar, 16, 3> control_grid;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                control_grid.row(i * 4 + j) << j, i, ((i + j) % 2 == 0) ? -1 : 1;
            }
        }
        patch.set_control_grid(control_grid);
        Eigen::Matrix<Scalar, 8, 1> knots_u, knots_v;
        knots_u << 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0;
        knots_v << 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 2.5, 2.5;
        patch.set_knots_u(knots_u);
        patch.set_knots_v(knots_v);

        Eigen::Matrix<Scalar, 16, 1> weights;
        weights.setOnes();
        SECTION("Uniform weight") { weights.setConstant(2.0); }
        SECTION("Non-uniform weight")
        {
            weights[5] = 2.0;
            weights[6] = 2.0;
            weights[9] = 2.0;
            weights[10] = 2.0;
        }
        patch.set_weights(weights);
        patch.initialize();

        validate_iso_curves(patch, 10);
        validate_derivative(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10);
        validate_inverse_evaluation_3d(patch, 10, 10);
    }
}

TEST_CASE("NURBSPatch 3", "[rational][nurbs_patch]")
{
    using namespace nanospline;
    using Scalar = double;

    SECTION("Cubic spline")
    {
        // BUG splitting NURBS patches can vary the size of the number of weights of
        // the resulting patches, which is templated to match the number of
        // control points. Needs -1, -1 for degree in order to compile.
        NURBSPatch<Scalar, 3, -1, -1> patch;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 3> control_grid(64, 3);
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                control_grid.row(i * 8 + j) << j, i, ((i + j) % 2 == 0) ? -1 : 1;
            }
        }
        patch.set_control_grid(control_grid);

        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> knots_u(12, 1);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> knots_v(12, 1);

        knots_u << 0.0, 0.0, 0.0, 0.0, .25, .5, .75, .9, 1.0, 1.0, 1.0, 1.0;
        knots_v << 0.0, 0.0, 0.0, 0.0, .25, 1., 1.5, 1.9, 2.5, 2.5, 2.5, 2.5;

        patch.set_knots_u(knots_u);
        patch.set_knots_v(knots_v);

        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> weights(64, 1);
        weights.setOnes();
        SECTION("Uniform weight") { weights.setConstant(2.0); }
        SECTION("Non-uniform weight")
        {
            std::mt19937 generator(0);
            std::uniform_int_distribution<int> dist(0, 63);
            for (int i = 0; i < 20; i++) {
                weights[dist(generator)] = 2.;
            }
        }
        SECTION("Zero weights")
        {
            weights.setConstant(1.0);
            weights[10] = 1e-15;
            weights[40] = 1e-15;
        }

        patch.set_weights(weights);
        patch.set_degree_u(3);
        patch.set_degree_v(3);
        patch.initialize();

        REQUIRE(patch.get_degree_u() == 3);
        REQUIRE(patch.get_degree_v() == 3);

        validate_iso_curves(patch, 10);
        validate_derivative(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10);
        validate_inverse_evaluation_3d(patch, 10, 10);
    }
}

TEST_CASE("NURBSPatch 4", "[rational][nurbs_patch]")
{
    using namespace nanospline;
    using Scalar = double;

    SECTION("Mixed degree")
    {
        NURBSPatch<Scalar, 3, -1, -1> patch;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 3> control_grid(14, 3);
        control_grid << 31.75, 0.0, 12.700000000000001, 31.75, 0.0, 0.0, 31.75, -54.99261314031184,
            12.700000000000001, 31.75, -54.99261314031184, 0.0, -15.874999999999993,
            -27.49630657015593, 12.700000000000001, -15.874999999999993, -27.49630657015593, 0.0,
            -63.499999999999986, -7.776507174585691e-15, 12.700000000000001, -63.499999999999986,
            -7.776507174585691e-15, 0.0, -15.875000000000014, 27.49630657015592, 12.700000000000001,
            -15.875000000000014, 27.49630657015592, 0.0, 31.74999999999995, 54.99261314031187,
            12.700000000000001, 31.74999999999995, 54.99261314031187, 0.0, 31.75,
            7.776507174585693e-15, 12.700000000000001, 31.75, 7.776507174585693e-15, 0.0;
        patch.set_control_grid(control_grid);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> weights(14, 1);
        weights << 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0;
        patch.set_weights(weights);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> u_knots(10, 1);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> v_knots(4, 1);
        u_knots << 0.0, 0.0, 0.0, 2.0943951023931953, 2.0943951023931953, 4.1887902047863905,
            4.1887902047863905, 6.283185307179586, 6.283185307179586, 6.283185307179586;
        v_knots << -10.573884999451131, -10.573884999451131, 2.12611500054887, 2.12611500054887;
        patch.set_knots_u(u_knots);
        patch.set_knots_v(v_knots);
        patch.set_degree_u(2);
        patch.set_degree_v(1);
        patch.initialize();

        REQUIRE(patch.get_degree_u() == 2);
        REQUIRE(patch.get_degree_v() == 1);
        validate_iso_curves(patch, 10);
        validate_derivative(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10);
    }
}

TEST_CASE("NURBSPatch 5", "[rational][nurbs_patch]")
{
    using namespace nanospline;
    using Scalar = double;

    SECTION("Extrapolation")
    {
        NURBSPatch<Scalar, 3, 3, 3> patch;
        Eigen::Matrix<Scalar, 16, 3> control_grid;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                control_grid.row(i * 4 + j) << j, i, ((i + j) % 2 == 0) ? -1 : 1;
            }
        }
        patch.set_control_grid(control_grid);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> u_knots(8, 1);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> v_knots(8, 1);
        u_knots << 0, 0, 0, 0, 1, 1, 1, 1;
        v_knots << 0, 0, 0, 0, 2, 2, 2, 2;
        patch.set_knots_u(u_knots);
        patch.set_knots_v(v_knots);

        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> weights(16, 1);
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
}

TEST_CASE("NURBSPatch 6", "[rational][nurbs_patch]")
{
    using namespace nanospline;
    using Scalar = double;

    SECTION("Inverse evaluation bug")
    {
        NURBSPatch<Scalar, 3, 2, 1> patch;
        Eigen::Matrix<Scalar, 14, 3> control_grid;
        control_grid << -760.0, -17.500000000000114, 27.5, 180.0, -17.5, 27.5, -760.0,
            -17.500000000000114, -2.810889132455344, 180.0, -17.5, -2.810889132455344, -760.0,
            8.749999999999881, 12.344555433772321, 180.0, 8.749999999999996, 12.344555433772321,
            -760.0, 34.99999999999988, 27.499999999999996, 180.0, 34.99999999999999,
            27.499999999999996, -760.0, 8.749999999999892, 42.65544456622767, 180.0,
            8.750000000000007, 42.65544456622767, -760.0, -17.500000000000085, 57.81088913245536,
            180.0, -17.49999999999997, 57.81088913245536, -760.0, -17.500000000000114,
            27.500000000000004, 180.0, -17.5, 27.500000000000004;
        patch.set_control_grid(control_grid);

        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> u_knots(10, 1);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> v_knots(4, 1);
        u_knots << 0.0, 0.0, 0.0, 2.0943951023931953, 2.0943951023931953, 4.1887902047863905,
            4.1887902047863905, 6.283185307179586, 6.283185307179586, 6.283185307179586;
        v_knots << -940.0, -940.0, 0.0, 0.0;
        patch.set_knots_u(u_knots);
        patch.set_knots_v(v_knots);

        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> weights(14, 1);
        weights << 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0;
        patch.set_weights(weights);
        patch.initialize();

        Eigen::Matrix<Scalar, 1, 3> bbox_min = control_grid.colwise().minCoeff();
        Eigen::Matrix<Scalar, 1, 3> bbox_max = control_grid.colwise().maxCoeff();
        const auto u_min = patch.get_u_lower_bound();
        const auto u_max = patch.get_u_upper_bound();
        const auto v_min = patch.get_v_lower_bound();
        const auto v_max = patch.get_v_upper_bound();

        Eigen::Matrix<Scalar, 4, 3> query_pts;
        query_pts << -572.86304534676469, -14.792976390831898, 36.420042475361221,
            -572.24631913639826, -16.181431755259489, 34.162412902005116, -569.7565428640977,
            -14.734586851799898, 36.56185691330802, -571.62196911575359, -15.23633166596376,
            35.714770763558114;

        for (int i = 0; i < 4; i++) {
            Eigen::Matrix<Scalar, 1, 3> p = query_pts.row(i);
            Eigen::Matrix<Scalar, 1, 2> p_uv =
                std::get<0>(patch.inverse_evaluate(p, u_min, u_max, v_min, v_max));
            Eigen::Matrix<Scalar, 1, 3> q = patch.evaluate(p_uv[0], p_uv[1]);

            REQUIRE((p - q).norm() < (bbox_max - bbox_min).norm() * 1e-3);
        }
    }
}

TEST_CASE("NURBSPatch Benchmark", "[!benchmark][numbs_patch]")
{
    using namespace nanospline;
    using Scalar = double;
    NURBSPatch<Scalar, 3, 3, 3> patch;
    Eigen::Matrix<Scalar, 16, 3> control_grid;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            control_grid.row(i * 4 + j) << j, i, ((i + j) % 2 == 0) ? -1 : 1;
        }
    }
    patch.set_control_grid(control_grid);
    Eigen::Matrix<Scalar, 8, 1> knots_u, knots_v;
    knots_u << 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0;
    knots_v << 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 2.5, 2.5;
    patch.set_knots_u(knots_u);
    patch.set_knots_v(knots_v);

    Eigen::Matrix<Scalar, 16, 1> weights;
    weights.setOnes();
    weights[5] = 2.0;
    weights[6] = 2.0;
    weights[9] = 2.0;
    weights[10] = 2.0;
    patch.set_weights(weights);
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
        Eigen::Matrix<Scalar, 3, 3> hessian;
        hessian.row(0) = patch.evaluate_2nd_derivative_uu(0.5, 0.6);
        hessian.row(1) = patch.evaluate_2nd_derivative_vv(0.5, 0.6);
        hessian.row(2) = patch.evaluate_2nd_derivative_uv(0.5, 0.6);
        return hessian;
    };
}

TEST_CASE("Periodic_debug", "[perioidc][nurbs]")
{
    using namespace nanospline;
    using Scalar = double;

    NURBSPatch<Scalar, 3, -1, -1> patch;
    Eigen::Matrix<Scalar, 14, 3> control_grid;
    control_grid << -12.7, 7.61831, 24.2906, 12.7, 7.61831, 24.2906, -12.7, 7.34012, 37.4859, 12.7,
        7.34012, 37.4859, -12.7, -3.94825, 30.6473, 12.7, -3.94825, 30.6473, -12.7, -15.2366,
        23.8088, 12.7, -15.2366, 23.8088, -12.7, -3.67006, 17.452, 12.7, -3.67006, 17.452, -12.7,
        7.89649, 11.0953, 12.7, 7.89649, 11.0953, -12.7, 7.61831, 24.2906, 12.7, 7.61831, 24.2906;

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> weights(14);
    weights << 1, 1, 0.5, 0.5, 1, 1, 0.5, 0.5, 1, 1, 0.5, 0.5, 1, 1;

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> knots_u(10);
    knots_u << -12.2182, -12.2182, -12.2182, -10.1238, -10.1238, -8.0294, -8.0294, -5.93501,
        -5.93501, -5.93501;

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> knots_v(4);
    knots_v << -27.6225, -27.6225, -2.2225, -2.2225;

    patch.set_control_grid(control_grid);
    patch.set_knots_u(knots_u);
    patch.set_knots_v(knots_v);
    patch.set_weights(weights);
    patch.set_degree_u(2);
    patch.set_degree_v(1);
    patch.set_periodic_u(true);
    patch.initialize();

    auto periodic_check = [](auto& patch, int num_samples = 10) {
        const Scalar u_min = patch.get_u_lower_bound();
        const Scalar u_max = patch.get_u_upper_bound();
        const Scalar v_min = patch.get_v_lower_bound();
        const Scalar v_max = patch.get_v_upper_bound();
        const Scalar u_period = u_max - u_min;

        for (int i = 0; i <= num_samples; i++) {
            for (int j = 0; j <= num_samples; j++) {
                Scalar u = u_min + (u_max - u_min) * i / (Scalar)num_samples;
                Scalar v = v_min + (v_max - v_min) * i / (Scalar)num_samples;
                auto p0 = patch.evaluate(u, v);
                auto p1 = patch.evaluate(u + u_period, v);
                auto p2 = patch.evaluate(u - u_period, v);
                REQUIRE((p0 - p1).norm() == Approx(0).margin(1e-6));
                REQUIRE((p0 - p2).norm() == Approx(0).margin(1e-6));
            }
        }
    };

    SECTION("Point check") { periodic_check(patch); }

    SECTION("Copy")
    {
        auto patch_copy = patch;
        REQUIRE(patch_copy.get_periodic_u());
        REQUIRE(!patch_copy.get_periodic_v());
        periodic_check(patch_copy);
    }

    SECTION("Clone")
    {
        auto patch_clone = patch.clone();
        REQUIRE(patch_clone->get_periodic_u());
        REQUIRE(!patch_clone->get_periodic_v());
        periodic_check(*patch_clone);
    }
}

TEST_CASE("Periodic_debug_2", "[periodic][numbs_patch]")
{
    using namespace nanospline;
    using Scalar = double;

    NURBSPatch<Scalar, 3, -1, -1> patch;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 3> control_grid(28, 3);
    control_grid << 12.7, 6.35, 31.3416, 10.9942, 6.35, 31.3416, 10.9942, 6.35, 33.0474, 12.7, 6.35,
        33.0474, 14.4058, 6.35, 33.0474, 14.4058, 6.35, 31.3416, 12.7, 6.35, 31.3416, 12.7, 6.35,
        29.6707, 7.65245, 6.35, 29.6707, 7.65245, 6.35, 34.7183, 12.7, 6.35, 34.7183, 17.7476, 6.35,
        34.7183, 17.7476, 6.35, 29.6707, 12.7, 6.35, 29.6707, 12.7, 5.38532, 28.3065, 4.92392,
        5.38532, 28.3065, 4.92392, 5.38532, 36.0825, 12.7, 5.38532, 36.0825, 20.4761, 5.38532,
        36.0825, 20.4761, 5.38532, 28.3065, 12.7, 5.38532, 28.3065, 12.7, 3.81, 27.7495, 3.81, 3.81,
        27.7495, 3.81, 3.81, 36.6395, 12.7, 3.81, 36.6395, 21.59, 3.81, 36.6395, 21.59, 3.81,
        27.7495, 12.7, 3.81, 27.7495;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> knots_u(8);
    knots_u << 6.91638e-08, 6.91638e-08, 6.91638e-08, 6.91638e-08, 6.28319, 6.28319, 6.28319,
        6.28319;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> knots_v(11);
    knots_v << 0.955317, 0.955317, 0.955317, 0.955317, 1.5708, 1.5708, 1.5708, 2.18628, 2.18628,
        2.18628, 2.18628;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> weights(28);
    weights << 1, 0.333333, 0.333333, 1, 0.333333, 0.333333, 1, 0.877664, 0.292555, 0.292555,
        0.877664, 0.292555, 0.292555, 0.877664, 0.877664, 0.292555, 0.292555, 0.877664, 0.292555,
        0.292555, 0.877664, 1, 0.333333, 0.333333, 1, 0.333333, 0.333333, 1;

    patch.set_control_grid(control_grid);
    patch.set_knots_u(knots_u);
    patch.set_knots_v(knots_v);
    patch.set_weights(weights);
    patch.set_degree_u(3);
    patch.set_degree_v(3);
    patch.set_periodic_v(true);
    patch.initialize();

    const Scalar u_min = patch.get_u_lower_bound();
    const Scalar u_max = patch.get_u_upper_bound();
    const Scalar v_min = patch.get_v_lower_bound();
    const Scalar v_max = patch.get_v_upper_bound();
    const Scalar u_delta = (u_max - u_min) / 5;
    const Scalar v_delta = (v_max - v_min) / 5;
    const Scalar prev_u = 1.38328e-07;
    const Scalar prev_v = 0.669671;

    Eigen::Matrix<double, 1, 3> q(13.4386, 6.35, 32.6209);
    auto uv = std::get<0>(patch.inverse_evaluate(
        q, prev_u - u_delta, prev_u + u_delta, prev_v - v_delta, prev_v + v_delta));
    Eigen::Matrix<double, 1, 3> p = patch.evaluate(uv[0], uv[1]);

    REQUIRE((q - p).norm() == Approx(0).margin(1e-6));
    REQUIRE(uv[0] == Approx(prev_u).margin(1e-3));
    REQUIRE(std::abs(uv[1] - prev_v) < 0.2);
}

TEST_CASE("Inverse_evaluate_debug", "[nurbs_patch][inverse_evaluation][singularity]")
{
    using namespace nanospline;
    using Scalar = double;

    NURBSPatch<Scalar, 3, -1, -1> patch;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 3> control_grid(9, 3);
    control_grid << 19.2165, -8.5725, 10.9601, 19.2165, -8.5725, 12.4841, 19.2165, -7.0485, 12.4841,
        19.2165, -8.5725, 10.9601, 22.0718, -8.5725, 12.4841, 22.0718, -7.0485, 12.4841, 19.2165,
        -8.5725, 10.9601, 20.4826, -8.5725, 10.1119, 20.4826, -7.0485, 10.1119;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> knots_u(6);
    knots_u << 4.71239, 4.71239, 4.71239, 6.87343, 6.87343, 6.87343;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> knots_v(6);
    knots_v << -1.5708, -1.5708, -1.5708, 0, 0, 0;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> weights(9);
    weights << 1, 0.707107, 1, 0.470868, 0.332954, 0.470868, 1, 0.707107, 1;

    patch.set_control_grid(control_grid);
    patch.set_knots_u(knots_u);
    patch.set_knots_v(knots_v);
    patch.set_weights(weights);
    patch.set_degree_u(2);
    patch.set_degree_v(2);
    patch.initialize();

    auto validate = [&](const auto& q, Scalar min_u, Scalar max_u, Scalar min_v, Scalar max_v) {
        auto uv = std::get<0>(patch.inverse_evaluate(q, min_u, max_u, min_v, max_v));
        REQUIRE(uv[0] >= min_u);
        REQUIRE(uv[0] <= max_u);
        REQUIRE(uv[1] >= min_v);
        REQUIRE(uv[1] <= max_v);

        constexpr Scalar eps = 1e-6;
        if (uv[0] > min_u + eps && uv[0] < max_u - eps && uv[1] > min_v + eps &&
            uv[1] < max_v - eps) {
            auto p = patch.evaluate(uv[0], uv[1]);
            REQUIRE((p - q).norm() < 1e-6);
        }
    };

    auto check_curve_projection = [&](auto curve) {
        auto samples = sample(curve, 10);
        for (auto t : samples) {
            auto p = curve.evaluate(t);
            validate(p,
                patch.get_u_lower_bound(),
                patch.get_u_upper_bound(),
                patch.get_v_lower_bound(),
                patch.get_v_upper_bound());
        }
    };

    SECTION("Sample 1")
    {
        Eigen::Matrix<Scalar, 1, 3> p(19.234620905067995, -8.5723435895248201, 10.947948027288934);
        Scalar min_u = 4.7123889803846879;
        Scalar max_u = 4.9284934154767051;
        Scalar min_v = -1.5707963267948966;
        Scalar max_v = -1.4137166941154069;
        validate(p, min_u, max_u, min_v, max_v);
    }

    SECTION("Sample 2")
    {
        Eigen::Matrix<Scalar, 1, 3> p(19.835675738832879, -8.3778249004781031, 10.545294801612789);
        validate(p,
            patch.get_u_lower_bound(),
            patch.get_u_upper_bound(),
            patch.get_v_lower_bound(),
            patch.get_v_upper_bound());
    }

    SECTION("Batch")
    {
        Eigen::Matrix<Scalar, Eigen::Dynamic, 3, Eigen::RowMajor> pts(18, 3);
        pts << 19.847621068486124, -8.3696625160634959, 10.537290071933757, 19.811630218749425,
            -8.3936441595428803, 10.561400879151087, 19.775189107434223, -8.416100490592946,
            10.585813323317703, 19.738362645992495, -8.4370188104524733, 10.61048391928345,
            19.701215595691497, -8.4563915624868713, 10.635369282509167, 19.663812231954171,
            -8.4742162301906916, 10.660426353930045, 19.62621602403852, -8.490495199308814,
            10.685612614543189, 19.588489332895676, -8.5052355870670269, 10.71088628781702,
            19.550693129645612, -8.5184490418280951, 10.736206528288594, 19.512886736686692,
            -8.5301515167349393, 10.761533594998161, 19.475127593021917, -8.540363021068945,
            10.786829008700622, 19.437471044951092, -8.5491073531393678, 10.812055692083986,
            19.399970162853982, -8.5564118185342153, 10.837178092509046, 19.362675584383521,
            -8.5623069375084633, 10.862162287056583, 19.325635384007459, -8.5668261451694221,
            10.886976069923399, 19.288894968487522, -8.5700054879490057, 10.911589022442341,
            19.252496997572315, -8.5718833196373421, 10.935972566211355, 19.2164813289059,
            -8.5724999999999998, 10.960100000000001;

        Scalar min_u = 4.7123889803846879;
        Scalar max_u = 6.8734333313048612;
        Scalar min_v = -0.7;
        Scalar max_v = 0;

        for (int i = 0; i < 18; i++) {
            validate(pts.row(i).eval(), min_u, max_u, min_v, max_v);
        }
    }

    SECTION("Boundary curves 1")
    {
        NURBS<Scalar, 3, -1> curve;
        Eigen::Matrix<Scalar, 3, 3, Eigen::RowMajor> control_pts;
        control_pts << 19.2165, -7.0485, 12.4841, 22.0718, -7.0485, 12.4841, 20.4826, -7.0485,
            10.1119;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> knots(6);
        knots << 0, 0, 0, 2.16104, 2.16104, 2.16104;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> weights(3);
        weights << 1, 0.470868, 1;

        curve.set_control_points(control_pts);
        curve.set_knots(knots);
        curve.set_weights(weights);
        curve.initialize();

        check_curve_projection(curve);
    }

    SECTION("Boundary curves 2")
    {
        NURBS<Scalar, 3, -1> curve;
        Eigen::Matrix<Scalar, 3, 3, Eigen::RowMajor> control_pts;
        control_pts << 19.2165, -8.5725, 10.9601, 20.4826, -8.5725, 10.1119, 20.4826, -7.0485,
            10.1119;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> knots(6);
        knots << 4.71239, 4.71239, 4.71239, 6.28319, 6.28319, 6.28319;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> weights(3);
        weights << 1, 0.707107, 1;

        curve.set_control_points(control_pts);
        curve.set_knots(knots);
        curve.set_weights(weights);
        curve.initialize();

        check_curve_projection(curve);
    }

    SECTION("Boundary curves 3")
    {
        NURBS<Scalar, 3, -1> curve;
        Eigen::Matrix<Scalar, 3, 3, Eigen::RowMajor> control_pts;
        control_pts << 19.2165, -7.0485, 12.4841, 19.2165, -8.5725, 12.4841, 19.2165, -8.5725,
            10.9601;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> knots(6);
        knots << 3.14159, 3.14159, 3.14159, 4.71239, 4.71239, 4.71239;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> weights(3);
        weights << 1, 0.707107, 1;

        curve.set_control_points(control_pts);
        curve.set_knots(knots);
        curve.set_weights(weights);
        curve.initialize();

        check_curve_projection(curve);
    }
}

TEST_CASE("Spiral", "[nurbs_patch][inverse_evaluation]")
{
    using namespace nanospline;
    using Scalar = double;

    NURBSPatch<Scalar, 3, -1, -1> patch;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 3> control_pts(1099, 3);
    control_pts << 13.0613, -2.68662, 16.2116, 12.8708, -4.80836, 16.9315, 12.6803, -6.28589,
        18.6158, 12.4898, -7.76342, 20.3001, 12.2993, -8.20086, 22.4975, 12.1088, -8.63829, 24.6949,
        11.9183, -7.91841, 26.8166, 11.7278, -7.19853, 28.9384, 11.5373, -5.51423, 30.4159, 11.3468,
        -3.82993, 31.8934, 11.1563, -1.63252, 32.3309, 10.9658, 0.564894, 32.7683, 10.7753, 2.68662,
        32.0484, 10.5848, 4.80836, 31.3285, 10.3943, 6.28589, 29.6442, 10.2038, 7.76342, 27.9599,
        10.0133, 8.20086, 25.7625, 9.82285, 8.63829, 23.5651, 9.63235, 7.91841, 21.4434, 9.44185,
        7.19853, 19.3216, 9.25135, 5.51423, 17.8441, 9.06085, 3.82993, 16.3666, 8.87035, 1.63252,
        15.9291, 8.67985, -0.564894, 15.4917, 8.48935, -2.68662, 16.2116, 8.29885, -4.80836,
        16.9315, 8.10835, -6.28589, 18.6158, 7.91785, -7.76342, 20.3001, 7.72735, -8.20086, 22.4975,
        7.53685, -8.63829, 24.6949, 7.34635, -7.91841, 26.8166, 7.15585, -7.19853, 28.9384, 6.96535,
        -5.51423, 30.4159, 6.77485, -3.82993, 31.8934, 6.58435, -1.63252, 32.3309, 6.39385,
        0.564894, 32.7683, 6.20335, 2.68662, 32.0484, 6.01285, 4.80836, 31.3285, 5.82235, 6.28589,
        29.6442, 5.63185, 7.76342, 27.9599, 5.44135, 8.20086, 25.7625, 5.25085, 8.63829, 23.5651,
        5.06035, 7.91841, 21.4434, 4.86985, 7.19853, 19.3216, 4.67935, 5.51423, 17.8441, 4.48885,
        3.82993, 16.3666, 4.29835, 1.63252, 15.9291, 4.10785, -0.564894, 15.4917, 3.91735, -2.68662,
        16.2116, 3.72685, -4.80836, 16.9315, 3.53635, -6.28589, 18.6158, 3.34585, -7.76342, 20.3001,
        3.15535, -8.20086, 22.4975, 2.96485, -8.63829, 24.6949, 2.77435, -7.91841, 26.8166, 2.58385,
        -7.19853, 28.9384, 2.39335, -5.51423, 30.4159, 2.20285, -3.82993, 31.8934, 2.01235,
        -1.63252, 32.3309, 1.82185, 0.564894, 32.7683, 1.63135, 2.68662, 32.0484, 1.44085, 4.80836,
        31.3285, 1.25035, 6.28589, 29.6442, 1.05985, 7.76342, 27.9599, 0.869348, 8.20086, 25.7625,
        0.678848, 8.63829, 23.5651, 0.488348, 7.91841, 21.4434, 0.297848, 7.19853, 19.3216,
        0.107348, 5.51423, 17.8441, -0.0831522, 3.82993, 16.3666, -0.273652, 1.63252, 15.9291,
        -0.464152, -0.564894, 15.4917, -0.654652, -2.68662, 16.2116, -0.845152, -4.80836, 16.9315,
        -1.03565, -6.28589, 18.6158, -1.22615, -7.76342, 20.3001, -1.41665, -8.20086, 22.4975,
        -1.60715, -8.63829, 24.6949, -1.79765, -7.91841, 26.8166, -1.98815, -7.19853, 28.9384,
        -2.17865, -5.51423, 30.4159, -2.36915, -3.82993, 31.8934, -2.55965, -1.63252, 32.3309,
        -2.75015, 0.564894, 32.7683, -2.94065, 2.68662, 32.0484, -3.13115, 4.80836, 31.3285,
        -3.32165, 6.28589, 29.6442, -3.51215, 7.76342, 27.9599, -3.70265, 8.20086, 25.7625,
        -3.89315, 8.63829, 23.5651, -4.08365, 7.91841, 21.4434, -4.27415, 7.19853, 19.3216,
        -4.46465, 5.51423, 17.8441, -4.65515, 3.82993, 16.3666, -4.84565, 1.63252, 15.9291,
        -5.03615, -0.564894, 15.4917, -5.22665, -2.68662, 16.2116, -5.41715, -4.80836, 16.9315,
        -5.60765, -6.28589, 18.6158, -5.79815, -7.76342, 20.3001, -5.98865, -8.20086, 22.4975,
        -6.17915, -8.63829, 24.6949, -6.36965, -7.91841, 26.8166, -6.56015, -7.19853, 28.9384,
        -6.75065, -5.51423, 30.4159, -6.94115, -3.82993, 31.8934, -7.13165, -1.63252, 32.3309,
        -7.32215, 0.564894, 32.7683, -7.51265, 2.68662, 32.0484, -7.70315, 4.80836, 31.3285,
        -7.89365, 6.28589, 29.6442, -8.08415, 7.76342, 27.9599, -8.27465, 8.20086, 25.7625,
        -8.46515, 8.63829, 23.5651, -8.65565, 7.91841, 21.4434, -8.84615, 7.19853, 19.3216,
        -9.03665, 5.51423, 17.8441, -9.22715, 3.82993, 16.3666, -9.41765, 1.63252, 15.9291,
        -9.60815, -0.564894, 15.4917, -9.79865, -2.68662, 16.2116, -9.98915, -4.80836, 16.9315,
        -10.1797, -6.28589, 18.6158, -10.3702, -7.76342, 20.3001, -10.5607, -8.20086, 22.4975,
        -10.7512, -8.63829, 24.6949, -10.9417, -7.91841, 26.8166, -11.1322, -7.19853, 28.9384,
        -11.3227, -5.51423, 30.4159, -11.5132, -3.82993, 31.8934, -11.7037, -1.63252, 32.3309,
        -11.8942, 0.564894, 32.7683, -12.0847, 2.68662, 32.0484, -12.2752, 4.80836, 31.3285,
        -12.4657, 6.28589, 29.6442, -12.6562, 7.76342, 27.9599, -12.8467, 8.20086, 25.7625,
        -13.0372, 8.63829, 23.5651, -13.2277, 7.91841, 21.4434, -13.4182, 7.19853, 19.3216,
        -13.6087, 5.51423, 17.8441, -13.7992, 3.82993, 16.3666, -13.9897, 1.63252, 15.9291,
        -14.1802, -0.564894, 15.4917, -14.3707, -2.68662, 16.2116, -14.5612, -4.80836, 16.9315,
        -14.7517, -6.28589, 18.6158, -14.9422, -7.76342, 20.3001, -15.1327, -8.20086, 22.4975,
        -15.3232, -8.63829, 24.6949, -15.5137, -7.91841, 26.8166, -15.7042, -7.19853, 28.9384,
        -15.8947, -5.51423, 30.4159, -16.0852, -3.82993, 31.8934, -16.2757, -1.63252, 32.3309,
        -16.4662, 0.564894, 32.7683, -16.6567, 2.68662, 32.0484, 13.4424, -2.35473, 17.1898,
        13.2519, -4.21435, 17.8207, 13.0614, -5.50936, 19.297, 12.8709, -6.80437, 20.7732, 12.6804,
        -7.18776, 22.6992, 12.4899, -7.57115, 24.6251, 12.2994, -6.94021, 26.4847, 12.1089,
        -6.30926, 28.3444, 11.9184, -4.83303, 29.6394, 11.7279, -3.3568, 30.9344, 11.5374, -1.43085,
        31.3178, 11.3469, 0.495109, 31.7012, 11.1564, 2.35473, 31.0702, 10.9659, 4.21435, 30.4393,
        10.7754, 5.50936, 28.963, 10.5849, 6.80437, 27.4868, 10.3944, 7.18776, 25.5608, 10.2039,
        7.57115, 23.6349, 10.0134, 6.94021, 21.7753, 9.82293, 6.30926, 19.9156, 9.63243, 4.83303,
        18.6206, 9.44193, 3.3568, 17.3256, 9.25143, 1.43085, 16.9422, 9.06093, -0.495109, 16.5588,
        8.87043, -2.35473, 17.1898, 8.67993, -4.21435, 17.8207, 8.48943, -5.50936, 19.297, 8.29893,
        -6.80437, 20.7732, 8.10843, -7.18776, 22.6992, 7.91793, -7.57115, 24.6251, 7.72743,
        -6.94021, 26.4847, 7.53693, -6.30926, 28.3444, 7.34643, -4.83303, 29.6394, 7.15593, -3.3568,
        30.9344, 6.96543, -1.43085, 31.3178, 6.77493, 0.495109, 31.7012, 6.58443, 2.35473, 31.0702,
        6.39393, 4.21435, 30.4393, 6.20343, 5.50936, 28.963, 6.01293, 6.80437, 27.4868, 5.82243,
        7.18776, 25.5608, 5.63193, 7.57115, 23.6349, 5.44143, 6.94021, 21.7753, 5.25093, 6.30926,
        19.9156, 5.06043, 4.83303, 18.6206, 4.86993, 3.3568, 17.3256, 4.67943, 1.43085, 16.9422,
        4.48893, -0.495109, 16.5588, 4.29843, -2.35473, 17.1898, 4.10793, -4.21435, 17.8207,
        3.91743, -5.50936, 19.297, 3.72693, -6.80437, 20.7732, 3.53643, -7.18776, 22.6992, 3.34593,
        -7.57115, 24.6251, 3.15543, -6.94021, 26.4847, 2.96493, -6.30926, 28.3444, 2.77443,
        -4.83303, 29.6394, 2.58393, -3.3568, 30.9344, 2.39343, -1.43085, 31.3178, 2.20293, 0.495109,
        31.7012, 2.01243, 2.35473, 31.0702, 1.82193, 4.21435, 30.4393, 1.63143, 5.50936, 28.963,
        1.44093, 6.80437, 27.4868, 1.25043, 7.18776, 25.5608, 1.05993, 7.57115, 23.6349, 0.869431,
        6.94021, 21.7753, 0.678931, 6.30926, 19.9156, 0.488431, 4.83303, 18.6206, 0.297931, 3.3568,
        17.3256, 0.107431, 1.43085, 16.9422, -0.0830685, -0.495109, 16.5588, -0.273569, -2.35473,
        17.1898, -0.464069, -4.21435, 17.8207, -0.654569, -5.50936, 19.297, -0.845069, -6.80437,
        20.7732, -1.03557, -7.18776, 22.6992, -1.22607, -7.57115, 24.6251, -1.41657, -6.94021,
        26.4847, -1.60707, -6.30926, 28.3444, -1.79757, -4.83303, 29.6394, -1.98807, -3.3568,
        30.9344, -2.17857, -1.43085, 31.3178, -2.36907, 0.495109, 31.7012, -2.55957, 2.35473,
        31.0702, -2.75007, 4.21435, 30.4393, -2.94057, 5.50936, 28.963, -3.13107, 6.80437, 27.4868,
        -3.32157, 7.18776, 25.5608, -3.51207, 7.57115, 23.6349, -3.70257, 6.94021, 21.7753,
        -3.89307, 6.30926, 19.9156, -4.08357, 4.83303, 18.6206, -4.27407, 3.3568, 17.3256, -4.46457,
        1.43085, 16.9422, -4.65507, -0.495109, 16.5588, -4.84557, -2.35473, 17.1898, -5.03607,
        -4.21435, 17.8207, -5.22657, -5.50936, 19.297, -5.41707, -6.80437, 20.7732, -5.60757,
        -7.18776, 22.6992, -5.79807, -7.57115, 24.6251, -5.98857, -6.94021, 26.4847, -6.17907,
        -6.30926, 28.3444, -6.36957, -4.83303, 29.6394, -6.56007, -3.3568, 30.9344, -6.75057,
        -1.43085, 31.3178, -6.94107, 0.495109, 31.7012, -7.13157, 2.35473, 31.0702, -7.32207,
        4.21435, 30.4393, -7.51257, 5.50936, 28.963, -7.70307, 6.80437, 27.4868, -7.89357, 7.18776,
        25.5608, -8.08407, 7.57115, 23.6349, -8.27457, 6.94021, 21.7753, -8.46507, 6.30926, 19.9156,
        -8.65557, 4.83303, 18.6206, -8.84607, 3.3568, 17.3256, -9.03657, 1.43085, 16.9422, -9.22707,
        -0.495109, 16.5588, -9.41757, -2.35473, 17.1898, -9.60807, -4.21435, 17.8207, -9.79857,
        -5.50936, 19.297, -9.98907, -6.80437, 20.7732, -10.1796, -7.18776, 22.6992, -10.3701,
        -7.57115, 24.6251, -10.5606, -6.94021, 26.4847, -10.7511, -6.30926, 28.3444, -10.9416,
        -4.83303, 29.6394, -11.1321, -3.3568, 30.9344, -11.3226, -1.43085, 31.3178, -11.5131,
        0.495109, 31.7012, -11.7036, 2.35473, 31.0702, -11.8941, 4.21435, 30.4393, -12.0846,
        5.50936, 28.963, -12.2751, 6.80437, 27.4868, -12.4656, 7.18776, 25.5608, -12.6561, 7.57115,
        23.6349, -12.8466, 6.94021, 21.7753, -13.0371, 6.30926, 19.9156, -13.2276, 4.83303, 18.6206,
        -13.4181, 3.3568, 17.3256, -13.6086, 1.43085, 16.9422, -13.7991, -0.495109, 16.5588,
        -13.9896, -2.35473, 17.1898, -14.1801, -4.21435, 17.8207, -14.3706, -5.50936, 19.297,
        -14.5611, -6.80437, 20.7732, -14.7516, -7.18776, 22.6992, -14.9421, -7.57115, 24.6251,
        -15.1326, -6.94021, 26.4847, -15.3231, -6.30926, 28.3444, -15.5136, -4.83303, 29.6394,
        -15.7041, -3.3568, 30.9344, -15.8946, -1.43085, 31.3178, -16.0851, 0.495109, 31.7012,
        -16.2756, 2.35473, 31.0702, 13.8235, -2.02284, 18.168, 13.633, -3.62035, 18.71, 13.4425,
        -4.73283, 19.9782, 13.252, -5.84531, 21.2463, 13.0615, -6.17467, 22.9008, 12.871, -6.50402,
        24.5553, 12.6805, -5.962, 26.1528, 12.49, -5.41999, 27.7504, 12.2995, -4.15183, 28.8628,
        12.109, -2.88367, 29.9753, 11.9185, -1.22917, 30.3047, 11.728, 0.425325, 30.634, 11.5375,
        2.02284, 30.092, 11.347, 3.62035, 29.55, 11.1565, 4.73283, 28.2818, 10.966, 5.84531,
        27.0137, 10.7755, 6.17467, 25.3592, 10.585, 6.50402, 23.7047, 10.3945, 5.962, 22.1072,
        10.204, 5.41999, 20.5096, 10.0135, 4.15183, 19.3972, 9.82302, 2.88367, 18.2847, 9.63252,
        1.22917, 17.9553, 9.44202, -0.425325, 17.626, 9.25152, -2.02284, 18.168, 9.06102, -3.62035,
        18.71, 8.87052, -4.73283, 19.9782, 8.68002, -5.84531, 21.2463, 8.48952, -6.17467, 22.9008,
        8.29902, -6.50402, 24.5553, 8.10852, -5.962, 26.1528, 7.91802, -5.41999, 27.7504, 7.72752,
        -4.15183, 28.8628, 7.53702, -2.88367, 29.9753, 7.34652, -1.22917, 30.3047, 7.15602,
        0.425325, 30.634, 6.96552, 2.02284, 30.092, 6.77502, 3.62035, 29.55, 6.58452, 4.73283,
        28.2818, 6.39402, 5.84531, 27.0137, 6.20352, 6.17467, 25.3592, 6.01302, 6.50402, 23.7047,
        5.82252, 5.962, 22.1072, 5.63202, 5.41999, 20.5096, 5.44152, 4.15183, 19.3972, 5.25102,
        2.88367, 18.2847, 5.06052, 1.22917, 17.9553, 4.87002, -0.425325, 17.626, 4.67952, -2.02284,
        18.168, 4.48902, -3.62035, 18.71, 4.29852, -4.73283, 19.9782, 4.10802, -5.84531, 21.2463,
        3.91752, -6.17467, 22.9008, 3.72702, -6.50402, 24.5553, 3.53652, -5.962, 26.1528, 3.34602,
        -5.41999, 27.7504, 3.15552, -4.15183, 28.8628, 2.96502, -2.88367, 29.9753, 2.77452,
        -1.22917, 30.3047, 2.58402, 0.425325, 30.634, 2.39352, 2.02284, 30.092, 2.20302, 3.62035,
        29.55, 2.01252, 4.73283, 28.2818, 1.82202, 5.84531, 27.0137, 1.63152, 6.17467, 25.3592,
        1.44102, 6.50402, 23.7047, 1.25052, 5.962, 22.1072, 1.06002, 5.41999, 20.5096, 0.869515,
        4.15183, 19.3972, 0.679015, 2.88367, 18.2847, 0.488515, 1.22917, 17.9553, 0.298015,
        -0.425325, 17.626, 0.107515, -2.02284, 18.168, -0.0829848, -3.62035, 18.71, -0.273485,
        -4.73283, 19.9782, -0.463985, -5.84531, 21.2463, -0.654485, -6.17467, 22.9008, -0.844985,
        -6.50402, 24.5553, -1.03548, -5.962, 26.1528, -1.22598, -5.41999, 27.7504, -1.41648,
        -4.15183, 28.8628, -1.60698, -2.88367, 29.9753, -1.79748, -1.22917, 30.3047, -1.98798,
        0.425325, 30.634, -2.17848, 2.02284, 30.092, -2.36898, 3.62035, 29.55, -2.55948, 4.73283,
        28.2818, -2.74998, 5.84531, 27.0137, -2.94048, 6.17467, 25.3592, -3.13098, 6.50402, 23.7047,
        -3.32148, 5.962, 22.1072, -3.51198, 5.41999, 20.5096, -3.70248, 4.15183, 19.3972, -3.89298,
        2.88367, 18.2847, -4.08348, 1.22917, 17.9553, -4.27398, -0.425325, 17.626, -4.46448,
        -2.02284, 18.168, -4.65498, -3.62035, 18.71, -4.84548, -4.73283, 19.9782, -5.03598,
        -5.84531, 21.2463, -5.22648, -6.17467, 22.9008, -5.41698, -6.50402, 24.5553, -5.60748,
        -5.962, 26.1528, -5.79798, -5.41999, 27.7504, -5.98848, -4.15183, 28.8628, -6.17898,
        -2.88367, 29.9753, -6.36948, -1.22917, 30.3047, -6.55998, 0.425325, 30.634, -6.75048,
        2.02284, 30.092, -6.94098, 3.62035, 29.55, -7.13148, 4.73283, 28.2818, -7.32198, 5.84531,
        27.0137, -7.51248, 6.17467, 25.3592, -7.70298, 6.50402, 23.7047, -7.89348, 5.962, 22.1072,
        -8.08398, 5.41999, 20.5096, -8.27448, 4.15183, 19.3972, -8.46498, 2.88367, 18.2847,
        -8.65548, 1.22917, 17.9553, -8.84598, -0.425325, 17.626, -9.03648, -2.02284, 18.168,
        -9.22698, -3.62035, 18.71, -9.41748, -4.73283, 19.9782, -9.60798, -5.84531, 21.2463,
        -9.79848, -6.17467, 22.9008, -9.98898, -6.50402, 24.5553, -10.1795, -5.962, 26.1528, -10.37,
        -5.41999, 27.7504, -10.5605, -4.15183, 28.8628, -10.751, -2.88367, 29.9753, -10.9415,
        -1.22917, 30.3047, -11.132, 0.425325, 30.634, -11.3225, 2.02284, 30.092, -11.513, 3.62035,
        29.55, -11.7035, 4.73283, 28.2818, -11.894, 5.84531, 27.0137, -12.0845, 6.17467, 25.3592,
        -12.275, 6.50402, 23.7047, -12.4655, 5.962, 22.1072, -12.656, 5.41999, 20.5096, -12.8465,
        4.15183, 19.3972, -13.037, 2.88367, 18.2847, -13.2275, 1.22917, 17.9553, -13.418, -0.425325,
        17.626, -13.6085, -2.02284, 18.168, -13.799, -3.62035, 18.71, -13.9895, -4.73283, 19.9782,
        -14.18, -5.84531, 21.2463, -14.3705, -6.17467, 22.9008, -14.561, -6.50402, 24.5553,
        -14.7515, -5.962, 26.1528, -14.942, -5.41999, 27.7504, -15.1325, -4.15183, 28.8628, -15.323,
        -2.88367, 29.9753, -15.5135, -1.22917, 30.3047, -15.704, 0.425325, 30.634, -15.8945,
        2.02284, 30.092, 14.2046, -1.69095, 19.1462, 14.0141, -3.02635, 19.5993, 13.8236, -3.9563,
        20.6594, 13.6331, -4.88626, 21.7195, 13.4426, -5.16157, 23.1025, 13.2521, -5.43689, 24.4855,
        13.0616, -4.9838, 25.8209, 12.8711, -4.53071, 27.1564, 12.6806, -3.47063, 28.0863, 12.4901,
        -2.41054, 29.0163, 12.2996, -1.0275, 29.2916, 12.1091, 0.355541, 29.5669, 11.9186, 1.69095,
        29.1138, 11.7281, 3.02635, 28.6607, 11.5376, 3.9563, 27.6006, 11.3471, 4.88626, 26.5405,
        11.1566, 5.16157, 25.1575, 10.9661, 5.43689, 23.7745, 10.7756, 4.9838, 22.4391, 10.5851,
        4.53071, 21.1036, 10.3946, 3.47063, 20.1737, 10.2041, 2.41054, 19.2437, 10.0136, 1.0275,
        18.9684, 9.8231, -0.355541, 18.6931, 9.6326, -1.69095, 19.1462, 9.4421, -3.02635, 19.5993,
        9.2516, -3.9563, 20.6594, 9.0611, -4.88626, 21.7195, 8.8706, -5.16157, 23.1025, 8.6801,
        -5.43689, 24.4855, 8.4896, -4.9838, 25.8209, 8.2991, -4.53071, 27.1564, 8.1086, -3.47063,
        28.0863, 7.9181, -2.41054, 29.0163, 7.7276, -1.0275, 29.2916, 7.5371, 0.355541, 29.5669,
        7.3466, 1.69095, 29.1138, 7.1561, 3.02635, 28.6607, 6.9656, 3.9563, 27.6006, 6.7751,
        4.88626, 26.5405, 6.5846, 5.16157, 25.1575, 6.3941, 5.43689, 23.7745, 6.2036, 4.9838,
        22.4391, 6.0131, 4.53071, 21.1036, 5.8226, 3.47063, 20.1737, 5.6321, 2.41054, 19.2437,
        5.4416, 1.0275, 18.9684, 5.2511, -0.355541, 18.6931, 5.0606, -1.69095, 19.1462, 4.8701,
        -3.02635, 19.5993, 4.6796, -3.9563, 20.6594, 4.4891, -4.88626, 21.7195, 4.2986, -5.16157,
        23.1025, 4.1081, -5.43689, 24.4855, 3.9176, -4.9838, 25.8209, 3.7271, -4.53071, 27.1564,
        3.5366, -3.47063, 28.0863, 3.3461, -2.41054, 29.0163, 3.1556, -1.0275, 29.2916, 2.9651,
        0.355541, 29.5669, 2.7746, 1.69095, 29.1138, 2.5841, 3.02635, 28.6607, 2.3936, 3.9563,
        27.6006, 2.2031, 4.88626, 26.5405, 2.0126, 5.16157, 25.1575, 1.8221, 5.43689, 23.7745,
        1.6316, 4.9838, 22.4391, 1.4411, 4.53071, 21.1036, 1.2506, 3.47063, 20.1737, 1.0601,
        2.41054, 19.2437, 0.869599, 1.0275, 18.9684, 0.679099, -0.355541, 18.6931, 0.488599,
        -1.69095, 19.1462, 0.298099, -3.02635, 19.5993, 0.107599, -3.9563, 20.6594, -0.0829012,
        -4.88626, 21.7195, -0.273401, -5.16157, 23.1025, -0.463901, -5.43689, 24.4855, -0.654401,
        -4.9838, 25.8209, -0.844901, -4.53071, 27.1564, -1.0354, -3.47063, 28.0863, -1.2259,
        -2.41054, 29.0163, -1.4164, -1.0275, 29.2916, -1.6069, 0.355541, 29.5669, -1.7974, 1.69095,
        29.1138, -1.9879, 3.02635, 28.6607, -2.1784, 3.9563, 27.6006, -2.3689, 4.88626, 26.5405,
        -2.5594, 5.16157, 25.1575, -2.7499, 5.43689, 23.7745, -2.9404, 4.9838, 22.4391, -3.1309,
        4.53071, 21.1036, -3.3214, 3.47063, 20.1737, -3.5119, 2.41054, 19.2437, -3.7024, 1.0275,
        18.9684, -3.8929, -0.355541, 18.6931, -4.0834, -1.69095, 19.1462, -4.2739, -3.02635,
        19.5993, -4.4644, -3.9563, 20.6594, -4.6549, -4.88626, 21.7195, -4.8454, -5.16157, 23.1025,
        -5.0359, -5.43689, 24.4855, -5.2264, -4.9838, 25.8209, -5.4169, -4.53071, 27.1564, -5.6074,
        -3.47063, 28.0863, -5.7979, -2.41054, 29.0163, -5.9884, -1.0275, 29.2916, -6.1789, 0.355541,
        29.5669, -6.3694, 1.69095, 29.1138, -6.5599, 3.02635, 28.6607, -6.7504, 3.9563, 27.6006,
        -6.9409, 4.88626, 26.5405, -7.1314, 5.16157, 25.1575, -7.3219, 5.43689, 23.7745, -7.5124,
        4.9838, 22.4391, -7.7029, 4.53071, 21.1036, -7.8934, 3.47063, 20.1737, -8.0839, 2.41054,
        19.2437, -8.2744, 1.0275, 18.9684, -8.4649, -0.355541, 18.6931, -8.6554, -1.69095, 19.1462,
        -8.8459, -3.02635, 19.5993, -9.0364, -3.9563, 20.6594, -9.2269, -4.88626, 21.7195, -9.4174,
        -5.16157, 23.1025, -9.6079, -5.43689, 24.4855, -9.7984, -4.9838, 25.8209, -9.9889, -4.53071,
        27.1564, -10.1794, -3.47063, 28.0863, -10.3699, -2.41054, 29.0163, -10.5604, -1.0275,
        29.2916, -10.7509, 0.355541, 29.5669, -10.9414, 1.69095, 29.1138, -11.1319, 3.02635,
        28.6607, -11.3224, 3.9563, 27.6006, -11.5129, 4.88626, 26.5405, -11.7034, 5.16157, 25.1575,
        -11.8939, 5.43689, 23.7745, -12.0844, 4.9838, 22.4391, -12.2749, 4.53071, 21.1036, -12.4654,
        3.47063, 20.1737, -12.6559, 2.41054, 19.2437, -12.8464, 1.0275, 18.9684, -13.0369,
        -0.355541, 18.6931, -13.2274, -1.69095, 19.1462, -13.4179, -3.02635, 19.5993, -13.6084,
        -3.9563, 20.6594, -13.7989, -4.88626, 21.7195, -13.9894, -5.16157, 23.1025, -14.1799,
        -5.43689, 24.4855, -14.3704, -4.9838, 25.8209, -14.5609, -4.53071, 27.1564, -14.7514,
        -3.47063, 28.0863, -14.9419, -2.41054, 29.0163, -15.1324, -1.0275, 29.2916, -15.3229,
        0.355541, 29.5669, -15.5134, 1.69095, 29.1138, 14.4983, -1.43519, 19.9, 14.3078, -2.56861,
        20.2846, 14.1173, -3.3579, 21.1843, 13.9268, -4.14719, 22.0841, 13.7363, -4.38087, 23.2579,
        13.5458, -4.61454, 24.4318, 13.3553, -4.22999, 25.5652, 13.1648, -3.84543, 26.6986, 12.9743,
        -2.94568, 27.4879, 12.7838, -2.04594, 28.2772, 12.5933, -0.872086, 28.5109, 12.4028,
        0.301764, 28.7445, 12.2123, 1.43519, 28.36, 12.0218, 2.56861, 27.9754, 11.8313, 3.3579,
        27.0757, 11.6408, 4.14719, 26.1759, 11.4503, 4.38087, 25.0021, 11.2598, 4.61454, 23.8282,
        11.0693, 4.22999, 22.6948, 10.8788, 3.84543, 21.5614, 10.6883, 2.94568, 20.7721, 10.4978,
        2.04594, 19.9828, 10.3073, 0.872086, 19.7491, 10.1168, -0.301764, 19.5155, 9.92627,
        -1.43519, 19.9, 9.73577, -2.56861, 20.2846, 9.54527, -3.3579, 21.1843, 9.35477, -4.14719,
        22.0841, 9.16427, -4.38087, 23.2579, 8.97377, -4.61454, 24.4318, 8.78327, -4.22999, 25.5652,
        8.59277, -3.84543, 26.6986, 8.40227, -2.94568, 27.4879, 8.21177, -2.04594, 28.2772, 8.02127,
        -0.872086, 28.5109, 7.83077, 0.301764, 28.7445, 7.64027, 1.43519, 28.36, 7.44977, 2.56861,
        27.9754, 7.25927, 3.3579, 27.0757, 7.06877, 4.14719, 26.1759, 6.87827, 4.38087, 25.0021,
        6.68777, 4.61454, 23.8282, 6.49727, 4.22999, 22.6948, 6.30677, 3.84543, 21.5614, 6.11627,
        2.94568, 20.7721, 5.92577, 2.04594, 19.9828, 5.73527, 0.872086, 19.7491, 5.54477, -0.301764,
        19.5155, 5.35427, -1.43519, 19.9, 5.16377, -2.56861, 20.2846, 4.97327, -3.3579, 21.1843,
        4.78277, -4.14719, 22.0841, 4.59227, -4.38087, 23.2579, 4.40177, -4.61454, 24.4318, 4.21127,
        -4.22999, 25.5652, 4.02077, -3.84543, 26.6986, 3.83027, -2.94568, 27.4879, 3.63977,
        -2.04594, 28.2772, 3.44927, -0.872086, 28.5109, 3.25877, 0.301764, 28.7445, 3.06827,
        1.43519, 28.36, 2.87777, 2.56861, 27.9754, 2.68727, 3.3579, 27.0757, 2.49677, 4.14719,
        26.1759, 2.30627, 4.38087, 25.0021, 2.11577, 4.61454, 23.8282, 1.92527, 4.22999, 22.6948,
        1.73477, 3.84543, 21.5614, 1.54427, 2.94568, 20.7721, 1.35377, 2.04594, 19.9828, 1.16327,
        0.872086, 19.7491, 0.972767, -0.301764, 19.5155, 0.782267, -1.43519, 19.9, 0.591767,
        -2.56861, 20.2846, 0.401267, -3.3579, 21.1843, 0.210767, -4.14719, 22.0841, 0.0202672,
        -4.38087, 23.2579, -0.170233, -4.61454, 24.4318, -0.360733, -4.22999, 25.5652, -0.551233,
        -3.84543, 26.6986, -0.741733, -2.94568, 27.4879, -0.932233, -2.04594, 28.2772, -1.12273,
        -0.872086, 28.5109, -1.31323, 0.301764, 28.7445, -1.50373, 1.43519, 28.36, -1.69423,
        2.56861, 27.9754, -1.88473, 3.3579, 27.0757, -2.07523, 4.14719, 26.1759, -2.26573, 4.38087,
        25.0021, -2.45623, 4.61454, 23.8282, -2.64673, 4.22999, 22.6948, -2.83723, 3.84543, 21.5614,
        -3.02773, 2.94568, 20.7721, -3.21823, 2.04594, 19.9828, -3.40873, 0.872086, 19.7491,
        -3.59923, -0.301764, 19.5155, -3.78973, -1.43519, 19.9, -3.98023, -2.56861, 20.2846,
        -4.17073, -3.3579, 21.1843, -4.36123, -4.14719, 22.0841, -4.55173, -4.38087, 23.2579,
        -4.74223, -4.61454, 24.4318, -4.93273, -4.22999, 25.5652, -5.12323, -3.84543, 26.6986,
        -5.31373, -2.94568, 27.4879, -5.50423, -2.04594, 28.2772, -5.69473, -0.872086, 28.5109,
        -5.88523, 0.301764, 28.7445, -6.07573, 1.43519, 28.36, -6.26623, 2.56861, 27.9754, -6.45673,
        3.3579, 27.0757, -6.64723, 4.14719, 26.1759, -6.83773, 4.38087, 25.0021, -7.02823, 4.61454,
        23.8282, -7.21873, 4.22999, 22.6948, -7.40923, 3.84543, 21.5614, -7.59973, 2.94568, 20.7721,
        -7.79023, 2.04594, 19.9828, -7.98073, 0.872086, 19.7491, -8.17123, -0.301764, 19.5155,
        -8.36173, -1.43519, 19.9, -8.55223, -2.56861, 20.2846, -8.74273, -3.3579, 21.1843, -8.93323,
        -4.14719, 22.0841, -9.12373, -4.38087, 23.2579, -9.31423, -4.61454, 24.4318, -9.50473,
        -4.22999, 25.5652, -9.69523, -3.84543, 26.6986, -9.88573, -2.94568, 27.4879, -10.0762,
        -2.04594, 28.2772, -10.2667, -0.872086, 28.5109, -10.4572, 0.301764, 28.7445, -10.6477,
        1.43519, 28.36, -10.8382, 2.56861, 27.9754, -11.0287, 3.3579, 27.0757, -11.2192, 4.14719,
        26.1759, -11.4097, 4.38087, 25.0021, -11.6002, 4.61454, 23.8282, -11.7907, 4.22999, 22.6948,
        -11.9812, 3.84543, 21.5614, -12.1717, 2.94568, 20.7721, -12.3622, 2.04594, 19.9828,
        -12.5527, 0.872086, 19.7491, -12.7432, -0.301764, 19.5155, -12.9337, -1.43519, 19.9,
        -13.1242, -2.56861, 20.2846, -13.3147, -3.3579, 21.1843, -13.5052, -4.14719, 22.0841,
        -13.6957, -4.38087, 23.2579, -13.8862, -4.61454, 24.4318, -14.0767, -4.22999, 25.5652,
        -14.2672, -3.84543, 26.6986, -14.4577, -2.94568, 27.4879, -14.6482, -2.04594, 28.2772,
        -14.8387, -0.872086, 28.5109, -15.0292, 0.301764, 28.7445, -15.2197, 1.43519, 28.36,
        15.3467, -1.43519, 19.9, 15.1562, -2.56861, 20.2846, 14.9657, -3.3579, 21.1843, 14.7752,
        -4.14719, 22.0841, 14.5847, -4.38087, 23.2579, 14.3942, -4.61454, 24.4318, 14.2037,
        -4.22999, 25.5652, 14.0132, -3.84543, 26.6986, 13.8227, -2.94568, 27.4879, 13.6322,
        -2.04594, 28.2772, 13.4417, -0.872086, 28.5109, 13.2512, 0.301764, 28.7445, 13.0607,
        1.43519, 28.36, 12.8702, 2.56861, 27.9754, 12.6797, 3.3579, 27.0757, 12.4892, 4.14719,
        26.1759, 12.2987, 4.38087, 25.0021, 12.1082, 4.61454, 23.8282, 11.9177, 4.22999, 22.6948,
        11.7272, 3.84543, 21.5614, 11.5367, 2.94568, 20.7721, 11.3462, 2.04594, 19.9828, 11.1557,
        0.872086, 19.7491, 10.9652, -0.301764, 19.5155, 10.7747, -1.43519, 19.9, 10.5842, -2.56861,
        20.2846, 10.3937, -3.3579, 21.1843, 10.2032, -4.14719, 22.0841, 10.0127, -4.38087, 23.2579,
        9.82223, -4.61454, 24.4318, 9.63173, -4.22999, 25.5652, 9.44123, -3.84543, 26.6986, 9.25073,
        -2.94568, 27.4879, 9.06023, -2.04594, 28.2772, 8.86973, -0.872086, 28.5109, 8.67923,
        0.301764, 28.7445, 8.48873, 1.43519, 28.36, 8.29823, 2.56861, 27.9754, 8.10773, 3.3579,
        27.0757, 7.91723, 4.14719, 26.1759, 7.72673, 4.38087, 25.0021, 7.53623, 4.61454, 23.8282,
        7.34573, 4.22999, 22.6948, 7.15523, 3.84543, 21.5614, 6.96473, 2.94568, 20.7721, 6.77423,
        2.04594, 19.9828, 6.58373, 0.872086, 19.7491, 6.39323, -0.301764, 19.5155, 6.20273,
        -1.43519, 19.9, 6.01223, -2.56861, 20.2846, 5.82173, -3.3579, 21.1843, 5.63123, -4.14719,
        22.0841, 5.44073, -4.38087, 23.2579, 5.25023, -4.61454, 24.4318, 5.05973, -4.22999, 25.5652,
        4.86923, -3.84543, 26.6986, 4.67873, -2.94568, 27.4879, 4.48823, -2.04594, 28.2772, 4.29773,
        -0.872086, 28.5109, 4.10723, 0.301764, 28.7445, 3.91673, 1.43519, 28.36, 3.72623, 2.56861,
        27.9754, 3.53573, 3.3579, 27.0757, 3.34523, 4.14719, 26.1759, 3.15473, 4.38087, 25.0021,
        2.96423, 4.61454, 23.8282, 2.77373, 4.22999, 22.6948, 2.58323, 3.84543, 21.5614, 2.39273,
        2.94568, 20.7721, 2.20223, 2.04594, 19.9828, 2.01173, 0.872086, 19.7491, 1.82123, -0.301764,
        19.5155, 1.63073, -1.43519, 19.9, 1.44023, -2.56861, 20.2846, 1.24973, -3.3579, 21.1843,
        1.05923, -4.14719, 22.0841, 0.868733, -4.38087, 23.2579, 0.678233, -4.61454, 24.4318,
        0.487733, -4.22999, 25.5652, 0.297233, -3.84543, 26.6986, 0.106733, -2.94568, 27.4879,
        -0.0837672, -2.04594, 28.2772, -0.274267, -0.872086, 28.5109, -0.464767, 0.301764, 28.7445,
        -0.655267, 1.43519, 28.36, -0.845767, 2.56861, 27.9754, -1.03627, 3.3579, 27.0757, -1.22677,
        4.14719, 26.1759, -1.41727, 4.38087, 25.0021, -1.60777, 4.61454, 23.8282, -1.79827, 4.22999,
        22.6948, -1.98877, 3.84543, 21.5614, -2.17927, 2.94568, 20.7721, -2.36977, 2.04594, 19.9828,
        -2.56027, 0.872086, 19.7491, -2.75077, -0.301764, 19.5155, -2.94127, -1.43519, 19.9,
        -3.13177, -2.56861, 20.2846, -3.32227, -3.3579, 21.1843, -3.51277, -4.14719, 22.0841,
        -3.70327, -4.38087, 23.2579, -3.89377, -4.61454, 24.4318, -4.08427, -4.22999, 25.5652,
        -4.27477, -3.84543, 26.6986, -4.46527, -2.94568, 27.4879, -4.65577, -2.04594, 28.2772,
        -4.84627, -0.872086, 28.5109, -5.03677, 0.301764, 28.7445, -5.22727, 1.43519, 28.36,
        -5.41777, 2.56861, 27.9754, -5.60827, 3.3579, 27.0757, -5.79877, 4.14719, 26.1759, -5.98927,
        4.38087, 25.0021, -6.17977, 4.61454, 23.8282, -6.37027, 4.22999, 22.6948, -6.56077, 3.84543,
        21.5614, -6.75127, 2.94568, 20.7721, -6.94177, 2.04594, 19.9828, -7.13227, 0.872086,
        19.7491, -7.32277, -0.301764, 19.5155, -7.51327, -1.43519, 19.9, -7.70377, -2.56861,
        20.2846, -7.89427, -3.3579, 21.1843, -8.08477, -4.14719, 22.0841, -8.27527, -4.38087,
        23.2579, -8.46577, -4.61454, 24.4318, -8.65627, -4.22999, 25.5652, -8.84677, -3.84543,
        26.6986, -9.03727, -2.94568, 27.4879, -9.22777, -2.04594, 28.2772, -9.41827, -0.872086,
        28.5109, -9.60877, 0.301764, 28.7445, -9.79927, 1.43519, 28.36, -9.98977, 2.56861, 27.9754,
        -10.1803, 3.3579, 27.0757, -10.3708, 4.14719, 26.1759, -10.5613, 4.38087, 25.0021, -10.7518,
        4.61454, 23.8282, -10.9423, 4.22999, 22.6948, -11.1328, 3.84543, 21.5614, -11.3233, 2.94568,
        20.7721, -11.5138, 2.04594, 19.9828, -11.7043, 0.872086, 19.7491, -11.8948, -0.301764,
        19.5155, -12.0853, -1.43519, 19.9, -12.2758, -2.56861, 20.2846, -12.4663, -3.3579, 21.1843,
        -12.6568, -4.14719, 22.0841, -12.8473, -4.38087, 23.2579, -13.0378, -4.61454, 24.4318,
        -13.2283, -4.22999, 25.5652, -13.4188, -3.84543, 26.6986, -13.6093, -2.94568, 27.4879,
        -13.7998, -2.04594, 28.2772, -13.9903, -0.872086, 28.5109, -14.1808, 0.301764, 28.7445,
        -14.3713, 1.43519, 28.36, 15.6404, -1.69095, 19.1462, 15.4499, -3.02635, 19.5993, 15.2594,
        -3.9563, 20.6594, 15.0689, -4.88626, 21.7195, 14.8784, -5.16157, 23.1025, 14.6879, -5.43689,
        24.4855, 14.4974, -4.9838, 25.8209, 14.3069, -4.53071, 27.1564, 14.1164, -3.47063, 28.0863,
        13.9259, -2.41054, 29.0163, 13.7354, -1.0275, 29.2916, 13.5449, 0.355541, 29.5669, 13.3544,
        1.69095, 29.1138, 13.1639, 3.02635, 28.6607, 12.9734, 3.9563, 27.6006, 12.7829, 4.88626,
        26.5405, 12.5924, 5.16157, 25.1575, 12.4019, 5.43689, 23.7745, 12.2114, 4.9838, 22.4391,
        12.0209, 4.53071, 21.1036, 11.8304, 3.47063, 20.1737, 11.6399, 2.41054, 19.2437, 11.4494,
        1.0275, 18.9684, 11.2589, -0.355541, 18.6931, 11.0684, -1.69095, 19.1462, 10.8779, -3.02635,
        19.5993, 10.6874, -3.9563, 20.6594, 10.4969, -4.88626, 21.7195, 10.3064, -5.16157, 23.1025,
        10.1159, -5.43689, 24.4855, 9.9254, -4.9838, 25.8209, 9.7349, -4.53071, 27.1564, 9.5444,
        -3.47063, 28.0863, 9.3539, -2.41054, 29.0163, 9.1634, -1.0275, 29.2916, 8.9729, 0.355541,
        29.5669, 8.7824, 1.69095, 29.1138, 8.5919, 3.02635, 28.6607, 8.4014, 3.9563, 27.6006,
        8.2109, 4.88626, 26.5405, 8.0204, 5.16157, 25.1575, 7.8299, 5.43689, 23.7745, 7.6394,
        4.9838, 22.4391, 7.4489, 4.53071, 21.1036, 7.2584, 3.47063, 20.1737, 7.0679, 2.41054,
        19.2437, 6.8774, 1.0275, 18.9684, 6.6869, -0.355541, 18.6931, 6.4964, -1.69095, 19.1462,
        6.3059, -3.02635, 19.5993, 6.1154, -3.9563, 20.6594, 5.9249, -4.88626, 21.7195, 5.7344,
        -5.16157, 23.1025, 5.5439, -5.43689, 24.4855, 5.3534, -4.9838, 25.8209, 5.1629, -4.53071,
        27.1564, 4.9724, -3.47063, 28.0863, 4.7819, -2.41054, 29.0163, 4.5914, -1.0275, 29.2916,
        4.4009, 0.355541, 29.5669, 4.2104, 1.69095, 29.1138, 4.0199, 3.02635, 28.6607, 3.8294,
        3.9563, 27.6006, 3.6389, 4.88626, 26.5405, 3.4484, 5.16157, 25.1575, 3.2579, 5.43689,
        23.7745, 3.0674, 4.9838, 22.4391, 2.8769, 4.53071, 21.1036, 2.6864, 3.47063, 20.1737,
        2.4959, 2.41054, 19.2437, 2.3054, 1.0275, 18.9684, 2.1149, -0.355541, 18.6931, 1.9244,
        -1.69095, 19.1462, 1.7339, -3.02635, 19.5993, 1.5434, -3.9563, 20.6594, 1.3529, -4.88626,
        21.7195, 1.1624, -5.16157, 23.1025, 0.971901, -5.43689, 24.4855, 0.781401, -4.9838, 25.8209,
        0.590901, -4.53071, 27.1564, 0.400401, -3.47063, 28.0863, 0.209901, -2.41054, 29.0163,
        0.0194012, -1.0275, 29.2916, -0.171099, 0.355541, 29.5669, -0.361599, 1.69095, 29.1138,
        -0.552099, 3.02635, 28.6607, -0.742599, 3.9563, 27.6006, -0.933099, 4.88626, 26.5405,
        -1.1236, 5.16157, 25.1575, -1.3141, 5.43689, 23.7745, -1.5046, 4.9838, 22.4391, -1.6951,
        4.53071, 21.1036, -1.8856, 3.47063, 20.1737, -2.0761, 2.41054, 19.2437, -2.2666, 1.0275,
        18.9684, -2.4571, -0.355541, 18.6931, -2.6476, -1.69095, 19.1462, -2.8381, -3.02635,
        19.5993, -3.0286, -3.9563, 20.6594, -3.2191, -4.88626, 21.7195, -3.4096, -5.16157, 23.1025,
        -3.6001, -5.43689, 24.4855, -3.7906, -4.9838, 25.8209, -3.9811, -4.53071, 27.1564, -4.1716,
        -3.47063, 28.0863, -4.3621, -2.41054, 29.0163, -4.5526, -1.0275, 29.2916, -4.7431, 0.355541,
        29.5669, -4.9336, 1.69095, 29.1138, -5.1241, 3.02635, 28.6607, -5.3146, 3.9563, 27.6006,
        -5.5051, 4.88626, 26.5405, -5.6956, 5.16157, 25.1575, -5.8861, 5.43689, 23.7745, -6.0766,
        4.9838, 22.4391, -6.2671, 4.53071, 21.1036, -6.4576, 3.47063, 20.1737, -6.6481, 2.41054,
        19.2437, -6.8386, 1.0275, 18.9684, -7.0291, -0.355541, 18.6931, -7.2196, -1.69095, 19.1462,
        -7.4101, -3.02635, 19.5993, -7.6006, -3.9563, 20.6594, -7.7911, -4.88626, 21.7195, -7.9816,
        -5.16157, 23.1025, -8.1721, -5.43689, 24.4855, -8.3626, -4.9838, 25.8209, -8.5531, -4.53071,
        27.1564, -8.7436, -3.47063, 28.0863, -8.9341, -2.41054, 29.0163, -9.1246, -1.0275, 29.2916,
        -9.3151, 0.355541, 29.5669, -9.5056, 1.69095, 29.1138, -9.6961, 3.02635, 28.6607, -9.8866,
        3.9563, 27.6006, -10.0771, 4.88626, 26.5405, -10.2676, 5.16157, 25.1575, -10.4581, 5.43689,
        23.7745, -10.6486, 4.9838, 22.4391, -10.8391, 4.53071, 21.1036, -11.0296, 3.47063, 20.1737,
        -11.2201, 2.41054, 19.2437, -11.4106, 1.0275, 18.9684, -11.6011, -0.355541, 18.6931,
        -11.7916, -1.69095, 19.1462, -11.9821, -3.02635, 19.5993, -12.1726, -3.9563, 20.6594,
        -12.3631, -4.88626, 21.7195, -12.5536, -5.16157, 23.1025, -12.7441, -5.43689, 24.4855,
        -12.9346, -4.9838, 25.8209, -13.1251, -4.53071, 27.1564, -13.3156, -3.47063, 28.0863,
        -13.5061, -2.41054, 29.0163, -13.6966, -1.0275, 29.2916, -13.8871, 0.355541, 29.5669,
        -14.0776, 1.69095, 29.1138;
    patch.set_control_grid(control_pts);

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> u_knots(11);
    u_knots << 0.59718, 0.59718, 0.59718, 0.59718, 0.877952, 0.877952, 0.877952, 1, 1, 1, 1;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> v_knots(160);
    v_knots << 0, 0, 0, 0.0119048, 0.0119048, 0.0238095, 0.0238095, 0.0357143, 0.0357143, 0.047619,
        0.047619, 0.0595238, 0.0595238, 0.0714286, 0.0714286, 0.0833333, 0.0833333, 0.0952381,
        0.0952381, 0.107143, 0.107143, 0.119048, 0.119048, 0.130952, 0.130952, 0.142857, 0.142857,
        0.154762, 0.154762, 0.166667, 0.166667, 0.178571, 0.178571, 0.190476, 0.190476, 0.202381,
        0.202381, 0.214286, 0.214286, 0.22619, 0.22619, 0.238095, 0.238095, 0.25, 0.25, 0.261905,
        0.261905, 0.27381, 0.27381, 0.285714, 0.285714, 0.297619, 0.297619, 0.309524, 0.309524,
        0.321429, 0.321429, 0.333333, 0.333333, 0.345238, 0.345238, 0.357143, 0.357143, 0.369048,
        0.369048, 0.380952, 0.380952, 0.392857, 0.392857, 0.404762, 0.404762, 0.416667, 0.416667,
        0.428571, 0.428571, 0.440476, 0.440476, 0.452381, 0.452381, 0.464286, 0.464286, 0.47619,
        0.47619, 0.488095, 0.488095, 0.5, 0.5, 0.511905, 0.511905, 0.52381, 0.52381, 0.535714,
        0.535714, 0.547619, 0.547619, 0.559524, 0.559524, 0.571429, 0.571429, 0.583333, 0.583333,
        0.595238, 0.595238, 0.607143, 0.607143, 0.619048, 0.619048, 0.630952, 0.630952, 0.642857,
        0.642857, 0.654762, 0.654762, 0.666667, 0.666667, 0.678571, 0.678571, 0.690476, 0.690476,
        0.702381, 0.702381, 0.714286, 0.714286, 0.72619, 0.72619, 0.738095, 0.738095, 0.75, 0.75,
        0.761905, 0.761905, 0.77381, 0.77381, 0.785714, 0.785714, 0.797619, 0.797619, 0.809524,
        0.809524, 0.821429, 0.821429, 0.833333, 0.833333, 0.845238, 0.845238, 0.857143, 0.857143,
        0.869048, 0.869048, 0.880952, 0.880952, 0.892857, 0.892857, 0.904762, 0.904762, 0.916667,
        0.916667, 0.928571, 0.928571, 0.928571;
    patch.set_knots_u(u_knots);
    patch.set_knots_v(v_knots);

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> weights(1099);
    weights << 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1,
        0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1,
        0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1,
        0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1,
        0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1,
        0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1,
        0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1,
        0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1,
        0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1,
        0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1,
        0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1,
        0.965926, 1, 0.965926, 1, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926,
        1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926,
        1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926,
        1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926,
        1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926,
        1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926,
        1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926,
        1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926,
        1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926,
        1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926,
        1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926,
        1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1,
        0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1,
        0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1,
        0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1,
        0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1,
        0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1,
        0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1,
        0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1,
        0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1,
        0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1,
        0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1,
        0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 1, 0.965926, 1, 0.965926,
        1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926,
        1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926,
        1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926,
        1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926,
        1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926,
        1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926,
        1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926,
        1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926,
        1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926,
        1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926,
        1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.564078,
        0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858,
        0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078,
        0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858,
        0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078,
        0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858,
        0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078,
        0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858,
        0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078,
        0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858,
        0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078,
        0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858,
        0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078,
        0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858,
        0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078,
        0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858,
        0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078,
        0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858,
        0.564078, 0.544858, 0.564078, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858,
        0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078,
        0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858,
        0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078,
        0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858,
        0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078,
        0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858,
        0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078,
        0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858,
        0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078,
        0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858,
        0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078,
        0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858,
        0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078,
        0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858,
        0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078,
        0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858,
        0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 0.544858, 0.564078, 1, 0.965926, 1,
        0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1,
        0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1,
        0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1,
        0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1,
        0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1,
        0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1,
        0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1,
        0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1,
        0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1,
        0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1,
        0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1, 0.965926, 1;
    patch.set_weights(weights);
    patch.set_degree_u(3);
    patch.set_degree_v(2);
    patch.initialize();

    const Scalar u_min = patch.get_u_lower_bound();
    const Scalar u_max = patch.get_u_upper_bound();
    const Scalar v_min = patch.get_v_lower_bound();
    const Scalar v_max = patch.get_v_upper_bound();

    SECTION("Sample 1")
    {
        Eigen::Matrix<Scalar, 1, 3> q(-12.7001, -2.05839, 19.8273);
        auto uv = std::get<0>(patch.inverse_evaluate(q, u_min, u_max, v_min, v_max));
        auto p = patch.evaluate(uv[0], uv[1]);
        REQUIRE((p - q).norm() < 1e-3);
    }

    SECTION("Sample 1 with incorrect range")
    {
        Eigen::Matrix<Scalar, 1, 3> q(-12.7001, -2.05839, 19.8273);
        auto uv = std::get<0>(patch.inverse_evaluate(q, 0.664387, 0.744951, 0.769556, 0.928571));
        auto p = patch.evaluate(uv[0], uv[1]);
        REQUIRE((p - q).norm() > 1);
    }
}

TEST_CASE("Inverse_eval_debug", "[inverse_evaluation][nurbs_patch]")
{
    using namespace nanospline;
    using Scalar = double;

    NURBSPatch<Scalar, 3, -1, -1> patch;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 3, Eigen::RowMajor> control_grid(14, 3);
    control_grid << -12.7, 7.61831, 24.2906, 12.7, 7.61831, 24.2906, -12.7, 7.34012, 37.4859, 12.7,
        7.34012, 37.4859, -12.7, -3.94825, 30.6473, 12.7, -3.94825, 30.6473, -12.7, -15.2366,
        23.8088, 12.7, -15.2366, 23.8088, -12.7, -3.67006, 17.452, 12.7, -3.67006, 17.452, -12.7,
        7.89649, 11.0953, 12.7, 7.89649, 11.0953, -12.7, 7.61831, 24.2906, 12.7, 7.61831, 24.2906;
    patch.set_control_grid(control_grid);

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> knots_u(10);
    knots_u << -12.2182, -12.2182, -12.2182, -10.1238, -10.1238, -8.0294, -8.0294, -5.93501,
        -5.93501, -5.93501;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> knots_v(4);
    knots_v << -27.6225, -27.6225, -2.2225, -2.2225;
    patch.set_knots_u(knots_u);
    patch.set_knots_v(knots_v);

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> weights(14);
    weights << 1, 1, 0.5, 0.5, 1, 1, 0.5, 0.5, 1, 1, 0.5, 0.5, 1, 1;
    patch.set_weights(weights);

    patch.set_periodic_u(true);
    patch.set_degree_u(2);
    patch.set_degree_v(1);
    patch.initialize();
    validate_derivative(patch, 10, 10, 1e-3);

    Eigen::Matrix<Scalar, 1, 3> q(-8.3820000000000014, 7.2159715908498701, 21.681703857772202);
    auto uv = std::get<0>(patch.inverse_evaluate(q,
        patch.get_u_lower_bound(),
        patch.get_u_upper_bound(),
        patch.get_v_lower_bound(),
        patch.get_v_upper_bound()));
    auto p = patch.evaluate(uv[0], uv[1]);

    REQUIRE((q - p).norm() == Approx(0).margin(1e-5));
}


TEST_CASE("Inverse evaluation singularity", "[inverse_evaluation][nurbs_patch]")
{
    using namespace nanospline;
    using Scalar = double;

    NURBSPatch<Scalar, 3, -1, -1> patch;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 3, Eigen::RowMajor> control_grid(28, 3);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> knots_u(11);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> knots_v(8);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> weights(28);

    control_grid << 59.5916771318613, -8.93962339061992e-15, -1.09478811709055e-30,
        59.5916771318613, -8.93962339061992e-15, -1.7879246781239798e-14, 59.5916771318613,
        8.93962339061992e-15, -1.7879246781239798e-14, 59.5916771318613, 8.93962339061992e-15, 0.0,
        59.2043514750084, 43.7716121739381, 5.36047647423326e-15, 59.2043514750084,
        43.7716121739381, 87.5432243478761, 59.2043514750084, -43.7716121739381, 87.5432243478761,
        59.2043514750084, -43.7716121739381, 0.0, 26.5125760205731, 82.22926481737811,
        1.00701805954842e-14, 26.5125760205731, 82.22926481737811, 164.458529634756,
        26.5125760205731, -82.22926481737811, 164.458529634756, 26.5125760205731,
        -82.22926481737811, 0.0, 2.0, 80.0, 9.79717439317883e-15, 2.0, 79.9999999999999, 160.0, 2.0,
        -80.0, 160.0, 2.0, -80.0, 0.0, 1.33333333333951, 79.9393708542237, 9.78974946424782e-15,
        1.33333333333951, 79.93937085422361, 159.878741708447, 1.33333333333951, -79.93937085422361,
        159.878741708447, 1.33333333333951, -79.9393708542237, 0.0, 0.666666666679028,
        79.87874170844731, 9.78232453531682e-15, 0.6666666666790291, 79.87874170844731,
        159.75748341689498, 0.6666666666790291, -79.87874170844731, 159.75748341689498,
        0.666666666679028, -79.87874170844731, 0.0, 1.85422423942816e-11, 79.818112562671,
        9.77489960638581e-15, 1.85422966043902e-11, 79.818112562671, 159.636225125342,
        1.85422966043902e-11, -79.818112562671, 159.636225125342, 1.85422423942816e-11,
        -79.818112562671, 0.0;
    knots_u << 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.02719692398306, 1.02719692398306,
        1.02719692398306, 1.02719692398306;
    knots_v << 0, 0, 0, 0, 1, 1, 1, 1;
    weights << 1.0, 0.333333333333334, 0.333333333333334, 1.0, 1.0, 0.333333333333334,
        0.333333333333334, 1.0, 1.0, 0.333333333333334, 0.333333333333334, 1.0, 1.0,
        0.333333333333334, 0.333333333333334, 1.0, 1.0, 0.333333333333334, 0.333333333333334, 1.0,
        1.0, 0.333333333333334, 0.333333333333334, 1.0, 1.0, 0.333333333333334, 0.333333333333334,
        1.0;

    patch.set_control_grid(control_grid);
    patch.set_knots_u(knots_u);
    patch.set_knots_v(knots_v);
    patch.set_weights(weights);
    patch.set_degree_u(3);
    patch.set_degree_v(3);
    patch.initialize();
    validate_derivative(patch, 10, 10, 1e-3);

    SECTION("inverse evaluate")
    {
        Eigen::Matrix<Scalar, 1, 3> q(44.530049812622238, -5.95954502452875, 50.13288085248999);
        auto uv = std::get<0>(patch.inverse_evaluate(
            q, 0.0102719692398306, 0.93560518468040033, 0.059533170036664362, 1));
        auto p = patch.evaluate(uv[0], uv[1]);

        REQUIRE((q - p).norm() == Approx(0).margin(1e-5));
    }

    SECTION("inverse evaluate 2")
    {
        Eigen::Matrix<Scalar, 1, 3> q(59.4612, 2.64846, 3.1301);
        auto uv = std::get<0>(patch.inverse_evaluate(
            q, patch.get_u_lower_bound() + 1e-3, patch.get_u_upper_bound(), 0.0, 1.0));
        auto p = patch.evaluate(uv[0], uv[1]);

        REQUIRE((q - p).norm() == Approx(0).margin(1e-3));
    }

    SECTION("inverse evaluate with initial guess")
    {
        Eigen::Matrix<Scalar, 1, 3> q(44.530049812622238, -5.95954502452875, 50.13288085248999);
        Eigen::Matrix<Scalar, 1, 2> uv;
        bool converged = false;
        std::tie(uv, converged) = patch.inverse_evaluate(q,
            0.5,
            0.5,
            patch.get_u_lower_bound(),
            patch.get_u_upper_bound(),
            patch.get_v_lower_bound(),
            patch.get_v_upper_bound());
        auto p = patch.evaluate(uv[0], uv[1]);

        REQUIRE(converged);
        REQUIRE((q - p).norm() == Approx(0).margin(1e-5));
    }
}

TEST_CASE("Inverse evaluation debug 2", "[inverse_evaluation][nurbs_patch]")
{
    using namespace nanospline;
    using Scalar = double;

    NURBSPatch<Scalar, 3, -1, -1> patch;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 3, Eigen::RowMajor> control_grid(126, 3);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> knots_u(46);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> knots_v(6);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> weights(126);

    control_grid << 45.963886292805, 12.3851671628894, 16.9031220303239, 45.9195259037511,
        12.376320292489, 16.1653378980796, 45.7683060209547, 13.0667277850766, 15.9489087451208,
        45.9506321147313, 12.3825238588476, 16.9047792887025, 45.8829392967195, 12.3683994544414,
        16.1653625446384, 45.71500074949169, 13.059186186078898, 15.9512728878924, 45.9376016548947,
        12.379646097631499, 16.907718190399702, 45.8470064955776, 12.3593482016871,
        16.1666189452838, 45.6625088997572, 13.0515527564067, 15.955935499148401, 45.9119569519479,
        12.3734546923284, 16.915980879512098, 45.776205729096795, 12.339097586126, 16.1713877437024,
        45.5590414649978, 13.0361170531216, 15.9695390583194, 45.8993427761901, 12.3701409924607,
        16.9213051290134, 45.741267471751705, 12.3278651418894, 16.1749172660507, 45.5080656838842,
        13.0283147260047, 15.978480864555802, 45.8745077169473, 12.363106100364801, 16.934217071478,
        45.6722011820683, 12.3032802332793, 16.1841189691954, 45.407547162122505, 13.0125622452188,
        16.0004309598062, 45.862286878753196, 12.359384865082, 16.9418050339176, 45.638008717528,
        12.289886929259, 16.1898164358191, 45.358004588425196, 13.0046120635138, 16.013439711728,
        45.84423823957761, 12.3535164125575, 16.954827433579602, 45.5872733778782, 12.2682404138947,
        16.1999119630346, 45.2847218657528, 12.9925967027485, 16.0359075346307, 45.8382691763997,
        12.3515124754218, 16.9594419777919, 45.5703826373583, 12.2607308661171, 16.2035535910736,
        45.2604663239373, 12.9885766176147, 16.0438898835253, 45.8293851471836, 12.3484371344149,
        16.966770076825, 45.5451609892038, 12.2490691344212, 16.2094058241384, 45.2243370478331,
        12.9825271680323, 16.0565964409092, 45.8264354010414, 12.3474004474369, 16.9692805105588,
        45.5367549423029, 12.245106786055201, 16.2114262438945, 45.2123363057423, 12.9805074669147,
        16.060954169072502, 45.820558939803, 12.3453042929394, 16.9744363507311, 45.519965488888204,
        12.2370423152757, 16.215600759699598, 45.188418966159, 12.9764622005982, 16.069913380332203,
        45.8176349261492, 12.3442457516903, 16.977079578797902, 45.5115889218823,
        12.232942879362799, 16.2177537086279, 45.1765133523985, 12.9744384669542, 16.0745110865087,
        45.803138670426996, 12.3389221138496, 16.9905669933528, 45.4697656117526, 12.2121014271824,
        16.2288486874253, 45.1174659051116, 12.9643534332795, 16.097994192900497, 45.7918864167085,
        12.3345463978496, 17.0022670399241, 45.4370449596765, 12.194586837943401, 16.2386468493123,
        45.0715567080695, 12.956357806245, 16.118436038209502, 45.7700093768281, 12.325599849945599,
        17.027347095459298, 45.372549605109, 12.1578396735143, 16.260038349971, 44.9821603078506,
        12.940539128154601, 16.1623916136318, 45.7593847476279, 12.3210289502155, 17.040727498435,
        45.3407401467285, 12.1385565097281, 16.271671819343698, 44.9386737050309, 12.9327160446605,
        16.1859060117074, 45.7387173449886, 12.3117272334079, 17.0690743972521, 45.2777835043985,
        12.098241078509, 16.2967590549673, 44.853950524156595, 12.9172789627739, 16.235853565927002,
        45.7286746664849, 12.306996359983899, 17.0840411114974, 45.2466030455318,
        12.077154165052502, 16.310257747424, 44.8127143088813, 12.9096649029096, 16.2622870768083,
        45.714022117792496, 12.2998055779242, 17.1076378406424, 45.2003241400474,
        12.044296687587899, 16.3318599939896, 44.7524545976817, 12.8984352860351, 16.3040630114296,
        45.709206602384796, 12.2973929009828, 17.1156947977574, 45.184877541009, 12.033060285558001,
        16.339341272890902, 44.7326341450625, 12.8947240685804, 16.3183431059646, 45.7020839419181,
        12.2937534656658, 17.128064217846898, 45.1618052997926, 12.015887070970699,
        16.3509210232046, 44.7032938617642, 12.8892103696775, 16.3402918274006, 45.6997271531914,
        12.2925372287563, 17.132233727368302, 45.154107015445, 12.010090832813699, 16.3548535980871,
        44.6935815541248, 12.8873818296993, 16.3476944674872, 45.6950629927637, 12.290107062033101,
        17.1406370744581, 45.1387750807406, 11.998418285501, 16.362822252869403, 44.6743527447681,
        12.8837560443747, 16.3626223583219, 45.692753534041, 12.288892071014502, 17.1448745080133,
        45.1311344499038, 11.992536221962, 16.366862421942002, 44.6648276468976, 12.8819571804528,
        16.3701539813137, 45.6813166829944, 12.2828188144956, 17.1662374667094, 45.0927846818757,
        11.9626952379443, 16.3874850873329, 44.617638405893096, 12.8730342048477,
        16.408145720567703, 45.672519909364304, 12.2779656286228, 17.1838892285511,
        45.0626708062588, 11.9382281564333, 16.404798122666797, 44.581280205156396,
        12.8661235738446, 16.4396045013929, 45.6555760273932, 12.26830204491, 17.220258416846,
        45.0028899413383, 11.8878206925649, 16.4413780143618, 44.5111358693159, 12.852796290428001,
        16.504567216440403, 45.6474290449154, 12.263491601660299, 17.2389759975471,
        44.973210307098405, 11.8618322820926, 16.4606968693475, 44.477350224045196,
        12.8463795889371, 16.5380713905334, 45.631745477105504, 12.2539467841312,
        17.277446967825497, 44.914033993055604, 11.8083139000969, 16.5015870084987,
        44.4122017327953, 12.8340921862082, 16.6070908820018, 45.624208959821, 12.249212361253099,
        17.297200417842, 44.8845227484615, 11.7807340832464, 16.5232184166218, 44.3808391538535,
        12.8282213968041, 16.6426062706032, 45.609720367768396, 12.2398557206596, 17.3377370158009,
        44.8254419279744, 11.723962799413199, 16.5691602671926, 44.320440339615295,
        12.8170930663531, 16.7156601928379, 45.6027683166554, 12.2352334449019, 17.3585201454218,
        44.795855650073, 11.6947198446356, 16.5935421531252, 44.291404201353195, 12.811835390747799,
        16.7531986519328, 45.5894346941608, 12.2261416711631, 17.4011331757126, 44.7364068814432,
        11.6345671735608, 16.6455747950658, 44.235609652407994, 12.8020186927523,
        16.830356642803398, 45.5830531052653, 12.2216720980023, 17.4229629769064, 44.7065258096736,
        11.6036052447699, 16.6733123742542, 44.208851181300695, 12.797459472518, 16.8699759439216,
        45.5708583060354, 12.212932675428501, 17.4677112683614, 44.646303715970696,
        11.5400024250649, 16.7328589729242, 44.15761096422359, 12.7891477758968, 16.9514037604537,
        45.565101742402696, 12.2087035351939, 17.4904230912154, 44.6162365013007,
        11.507621775188701, 16.7644891277929, 44.1333672531777, 12.785434454502301,
        16.9928364001478, 45.5542603322862, 12.2005770176171, 17.536474862012803, 44.5561471243601,
        11.4419837948178, 16.8320208152487, 44.0876027538521, 12.778984166036201,
        17.077077346759598, 45.5491731491669, 12.1966779029975, 17.559823457546, 44.5261276149784,
        11.4087056030748, 16.8680271285684, 44.0660721532033, 12.776245483179, 17.1199013601672,
        45.5396568859199, 12.1892548343086, 17.6071855633129, 44.4661398402078, 11.3415670137187,
        16.9453063279468, 44.0256859332067, 12.7718435141752, 17.2070198838102, 45.5352277550489,
        12.1857307982662, 17.631199204017, 44.4361956965535, 11.3077166485766, 16.986684730618098,
        44.0068301841443, 12.7701800783872, 17.2513143876148, 45.5311308463117, 12.182421768678601,
        17.6555621467287, 44.4067557723278, 11.274275987053999, 17.0307165320659, 43.9893285299962,
        12.7691175915198, 17.296391492798;

    knots_u << 0.00352667012400581, 0.00352667012400581, 0.00352667012400581, 0.00352667012400581,
        0.00360646810796613, 0.00360646810796613, 0.00368626609192646, 0.00368626609192646,
        0.00376606407588679, 0.00376606407588679, 0.00380596306786695, 0.00380596306786695,
        0.00382591256385703, 0.00382591256385703, 0.00384586205984711, 0.00384586205984711,
        0.00392566004380744, 0.00392566004380744, 0.00400545802776776, 0.00400545802776776,
        0.00408525601172809, 0.00408525601172809, 0.00412515500370825, 0.00412515500370825,
        0.00414510449969833, 0.00414510449969833, 0.00416505399568842, 0.00416505399568842,
        0.00424485197964874, 0.00424485197964874, 0.00432464996360907, 0.00432464996360907,
        0.0044044479475694, 0.0044044479475694, 0.00448424593152972, 0.00448424593152972,
        0.00456404391549005, 0.00456404391549005, 0.00464384189945037, 0.00464384189945037,
        0.0047236398834107, 0.0047236398834107, 0.00480343786737102, 0.00480343786737102,
        0.00480343786737102, 0.00480343786737102;
    knots_v << 0, 0, 0, 1, 1, 1;

    weights << 1.0, 0.804161128426488, 1.0, 1.0, 0.802973210189554, 1.0, 1.0, 0.801332445645098,
        1.0, 1.0, 0.797203984987235, 1.0, 1.0, 0.79471642165059, 1.0, 1.0, 0.788939612528814, 1.0,
        1.0, 0.785650272648997, 1.0, 1.0, 0.780139509705676, 1.0, 1.0, 0.778206375405051, 1.0, 1.0,
        0.775164719178105, 1.0, 1.0, 0.774127169299666, 1.0, 1.0, 0.77200508162923, 1.0, 1.0,
        0.770921450883741, 1.0, 1.0, 0.765413080994972, 1.0, 1.0, 0.760700102719825, 1.0, 1.0,
        0.750722269493806, 1.0, 1.0, 0.745457256955646, 1.0, 1.0, 0.734423484491492, 1.0, 1.0,
        0.728654611643831, 1.0, 1.0, 0.719653307285982, 1.0, 1.0, 0.716594815101432, 1.0, 1.0,
        0.711923318635199, 1.0, 1.0, 0.710352568132572, 1.0, 1.0, 0.707194913413766, 1.0, 1.0,
        0.705606648593325, 1.0, 1.0, 0.69762001838988, 1.0, 1.0, 0.691085911534018, 1.0, 1.0,
        0.677772434302037, 1.0, 1.0, 0.670992953541088, 1.0, 1.0, 0.65722967873, 1.0, 1.0,
        0.650245783277885, 1.0, 1.0, 0.636114426522641, 1.0, 1.0, 0.628966857710216, 1.0, 1.0,
        0.614551579933428, 1.0, 1.0, 0.607283742196491, 1.0, 1.0, 0.592677306459935, 1.0, 1.0,
        0.585406450450475, 1.0, 1.0, 0.571006180865696, 1.0, 1.0, 0.563873815729331, 1.0, 1.0,
        0.549814323834486, 1.0, 1.0, 0.542887301717408, 1.0, 1.0, 0.536104017477005, 1.0;

    patch.set_control_grid(control_grid);
    patch.set_knots_u(knots_u);
    patch.set_knots_v(knots_v);
    patch.set_weights(weights);
    patch.set_degree_u(3);
    patch.set_degree_v(2);
    patch.initialize();
    validate_derivative(patch, 10, 10, 1e-3);


    Eigen::Matrix<Scalar, 1, 3> q(45.7915, 12.3343, 17.003);
    Eigen::Matrix<Scalar, 1, 2> uv;
    bool converged = false;
    std::tie(uv, converged) = patch.inverse_evaluate(q,
        0.00352667,
        3.09622e-15,
        patch.get_u_lower_bound(),
        patch.get_u_upper_bound(),
        patch.get_v_lower_bound(),
        patch.get_v_upper_bound());

    // This failed simply because the initial guess is too far from the closest
    // point.
    REQUIRE(!converged);
}

TEST_CASE("Inverse evaluation debug 3", "[inverse_evaluation][nurbs_patch]")
{
    using namespace nanospline;
    using Scalar = double;

    NURBSPatch<Scalar, 3, -1, -1> patch;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 3, Eigen::RowMajor> control_grid(32, 3);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> knots_u(8);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> knots_v(12);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> weights(32);

    control_grid << -38.880008731, 8.385493702, 71.226919957, -39.402300098, 8.55255348,
        71.327128968, -40.320142808, 8.911711629000001, 71.503331809, -41.292045175,
        9.053877490000001, 71.68954345799999, -41.714626345999996, 9.137479717, 71.77054169200001,
        -41.724226387921306, 9.139378961308529, 71.7723817797638, -41.733826429842495,
        9.14127820561705, 71.77422186752749, -41.743426471763804, 9.14317744992557,
        71.77606195529131, -38.880008731, 10.291480272, 67.874078765, -39.402300098, 10.45854005,
        67.974287776, -40.320142808, 10.817698198999999, 68.1504906159999, -41.292045175,
        10.95986406, 68.336702265, -41.714626345999996, 11.043466287, 68.4177005,
        -41.724226387921306, 11.0453655313085, 68.4195405877864, -41.7338264298426,
        11.0472647756171, 68.42138067557289, -41.743426471763804, 11.0491640199256,
        68.4232207633595, -38.880008731, 10.205310773999999, 64.01824793, -39.402300098,
        10.372370552, 64.118456941, -40.320142808, 10.731528701, 64.2946597820001, -41.292045175,
        10.873694562, 64.48087143100011, -41.714626345999996, 10.957296789, 64.56186966499999,
        -41.724226387921306, 10.9591960333085, 64.5637097527637, -41.733826429842495,
        10.9610952776171, 64.5655498405275, -41.743426471763804, 10.9629945219256,
        64.56738992829129, -38.880008731, 8.151252515000001, 60.753859934000005, -39.402300098,
        8.318312292, 60.854068945, -40.320142808, 8.677470442, 61.0302717860001, -41.292045175,
        8.81963630200001, 61.216483435, -41.714626345999996, 8.903238529, 61.297481669,
        -41.724226387921306, 8.90513777330853, 61.299321756763696, -41.733826429842495,
        8.90703701761705, 61.301161844527506, -41.743426471763804, 8.908936261925579,
        61.303001932291195;


    knots_u << 0.525905430141845, 0.525905430141845, 0.525905430141845, 0.525905430141845,
        0.668840505851333, 0.668840505851333, 0.668840505851333, 0.668840505851333;

    knots_v << 0.0, 0.0, 0.0, 0.0, 0.559813202415503, 1.0, 1.0, 1.0, 1.01, 1.01, 1.01, 1.01;


    weights << 0.937208479557941, 0.937208479557941, 0.937208479557941, 0.937208479557941,
        0.937208479557941, 0.937208479557941, 0.937208479557941, 0.937208479557941,
        0.848511106177263, 0.848511106177263, 0.848511106177263, 0.848511106177263,
        0.848511106177263, 0.848511106177263, 0.848511106177263, 0.848511106177263,
        0.848481003425045, 0.848481003425045, 0.848481003425045, 0.848481003425046,
        0.848481003425046, 0.848481003425046, 0.848481003425046, 0.848481003425046,
        0.937118171301287, 0.937118171301287, 0.937118171301287, 0.937118171301288,
        0.937118171301288, 0.937118171301288, 0.937118171301288, 0.937118171301288;


    patch.set_control_grid(control_grid);
    patch.set_knots_u(knots_u);
    patch.set_knots_v(knots_v);
    patch.set_weights(weights);
    patch.set_degree_u(3);
    patch.set_degree_v(3);
    patch.initialize();
    validate_derivative(patch, 10, 10, 1e-3);


    Eigen::Matrix<Scalar, 1, 3> q(-41.714629087878905, 9.1767576153057, 71.700813713449307);
    Eigen::Matrix<Scalar, 1, 2> uv;
    bool converged = false;
    std::tie(uv, converged) = patch.inverse_evaluate(
        q, 0.52675287100000112, 0.66798487974635101, -1.0000000000000001e-15, 1.0000009575210089);

    REQUIRE(converged);
    auto p = patch.evaluate(uv[0], uv[1]);
    REQUIRE((q - p).norm() == Approx(0).margin(1e-5));
}

TEST_CASE("Inverse evaluation debug 4", "[inverse_evaluation][nurbs_patch]")
{
    using namespace nanospline;
    using Scalar = double;

    NURBSPatch<Scalar, 3, -1, -1> patch;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 3, Eigen::RowMajor> control_grid(91, 3);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> knots_u(14);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> knots_v(20);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> weights(91);

    control_grid << 314.464159205525, 211.693510927675, 1.3751634508362198, 314.469739041021,
        216.299864800432, 1.37386971257957, 313.36454779504203, 220.72611913729202,
        1.37342369342761, 311.277763499612, 224.777502810096, 1.3735590045091999, 308.338564186756,
        228.259244690688, 1.37400925695319, 304.6761278885, 230.97657365091, 1.37450806188845,
        300.419632636867, 232.734718562605, 1.37478903044384, 299.040269769884, 233.30446424834,
        1.37488008131297, 297.616243244955, 233.76502476644401, 1.3748054829874201,
        296.158553552253, 234.11385847508402, 1.3746727055802799, 294.67820118194896,
        234.348423732426, 1.37458921920461, 293.186186624216, 234.466178896635, 1.37466249397349,
        291.693510369227, 234.46458232588, 1.375, 314.53694450325696, 211.693510648435,
        1.6459305514128, 314.541594495522, 216.3142343283, 1.64485240620251, 313.432504023114,
        220.754240032144, 1.64440345071905, 311.339076844886, 224.81816342239, 1.64443275911361,
        308.390732356586, 228.310681512065, 1.64479067681018, 304.716854164304, 231.03651447252298,
        1.64532651064224, 300.44676428700603, 232.800354443221, 1.64588991411761,
        299.06302792421104, 233.37193246108302, 1.6460724889355, 297.63455444814105,
        233.834136867727, 1.64598601923679, 296.17235661563603, 234.184363430106, 1.64577461911166,
        294.68744500655805, 234.42000923508, 1.6455825999567701, 293.190827739321, 234.538472714605,
        1.64555412034986, 291.693510369227, 234.53715189883002, 1.64583333333329,
        314.60964267879297, 211.693510480898, 1.9167210589089299, 314.613362597992,
        216.328459657629, 1.91585856003882, 313.500093301167, 220.781925208788, 1.9155025988237302,
        311.39968129992, 224.858205661533, 1.91553117702756, 308.441987098239, 228.36163500339902,
        1.9158233395684, 304.756840403319, 231.096584315555, 1.91625750394182, 300.474018555788,
        232.867401584648, 1.91671211184508, 299.09318974298196, 233.438335490635, 1.91685896093922,
        297.665998960137, 233.90141113985598, 1.91678957886251, 296.202442021045,
        234.25335747815802, 1.91661975131993, 294.712513312619, 234.490904233192,
        1.9164654352206498, 293.20620564488996, 234.610782076235, 1.91644246088432,
        291.693510369228, 234.60972147178, 1.9166666666666299, 314.682275512363, 211.69351039713598,
        2.18752911542016, 314.68506535872496, 216.34261695725098, 2.18688226283986, 313.5674836274,
        220.809395205034, 2.18666800135138, 311.459927096426, 224.897938947693, 2.18674914123456,
        308.4927958803, 228.41234734719302, 2.18698855157528, 304.796485222117, 231.15672534119,
        2.18724955922163, 300.501383397018, 232.935176362077, 2.18739492146164, 299.12711044329296,
        233.504231806427, 2.18743877995311, 297.704068854978, 233.967785914629, 2.1874008509517897,
        296.2407227456, 234.32160414471102, 2.1873345569760896, 294.74553693676097,
        234.561451337928, 2.18729335060225, 293.22697713269105, 234.683092037525,
        2.1873306746539902, 291.693510369227, 234.68229104473002, 2.1875, 314.754864784197,
        211.693510369221, 2.4583488630420502, 314.756724657012, 216.356782396442, 2.45791763373108,
        313.63484603705297, 220.836878273656, 2.45784436121924, 311.520171535266,
        224.93769005890098, 2.45797661848685, 308.543635701688, 228.463083869413, 2.45816103990404,
        304.83619522288103, 231.21689899668, 2.4582457010629, 300.528846764503, 233.0029951507,
        2.4580776408027196, 299.161128566856, 233.57013960721702, 2.4580245543992603,
        297.742236189568, 234.03415467794, 2.45801978012254, 296.27909674356, 234.389838539166,
        2.4580558293676904, 294.77864002544004, 234.63198748890397, 2.4581250957617398,
        293.247798977807, 234.75539635174, 2.4582200821835003, 291.693510369228, 234.75486061768,
        2.45833333333336, 314.82743227452596, 211.69351036922802, 2.7291744438701597,
        314.828362247109, 216.371032144701, 2.7289588208285, 313.702353871959, 220.864608600736,
        2.7289747197635497, 311.58077733219096, 224.977799122739, 2.7290997847962304,
        308.594990881201, 228.514127695245, 2.7292104541147197, 304.876383149373, 231.277081561892,
        2.72918471312237, 300.556396612047, 233.070174325708, 2.7288995677037, 299.191614202087,
        233.636653280468, 2.72880947949939, 297.77401087607603, 234.101505347575, 2.72882237981819,
        296.309490359471, 234.458871103011, 2.7289014604003, 294.80395934146, 234.702889117552,
        2.72900974907698, 293.26332795337703, 234.827696285756, 2.72911041566229, 291.693510369227,
        234.82743019063, 2.7291666666667, 314.89999976358, 211.69351036922802, 3.00000000000004,
        314.89999976358, 216.385442371575, 3.0, 313.770183458059, 220.89282792630598, 3.0,
        311.64211413265, 225.018623263081, 3.0, 308.64735507299, 228.56578461156198, 3.0,
        304.917469564714, 231.337268201413, 3.0, 300.584020893456, 233.136030262293, 3.0,
        299.214920835337, 233.704327115196, 3.0, 297.79288290650703, 234.170780386034, 3.0,
        296.32381500122403, 234.529482180549, 3.0, 294.813625013744, 234.77452460448401, 3.0,
        293.268220838326, 234.89999976358, 3.0, 291.693510369227, 234.89999976358, 3.0;

    knots_u << 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2;

    knots_v << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0,
        3.0, 3.0, 3.0;

    weights.setOnes();

    patch.set_control_grid(control_grid);
    patch.set_knots_u(knots_u);
    patch.set_knots_v(knots_v);
    patch.set_weights(weights);
    patch.set_degree_u(6);
    patch.set_degree_v(6);
    patch.initialize();

    auto test_inverse_evaluate = [&](const auto q) {
        Eigen::Matrix<Scalar, 1, 2> uv;
        bool converged = false;
        std::tie(uv, converged) = patch.inverse_evaluate(q, 0.999999999999997, 2, 1, 3);
        auto p = patch.evaluate(uv[0], uv[1]);

        REQUIRE(converged);
        REQUIRE((q - p).norm() == Approx(0).margin(1e-5));
    };

    SECTION("Query 1")
    {
        Eigen::Matrix<Scalar, 1, 3> q(303.22473803891199, 231.34315979588698, 1.37456101266505);
        test_inverse_evaluate(q);
    }
    SECTION("Query 2")
    {
        Eigen::Matrix<Scalar, 1, 3> q(303.3317680507036, 231.2801257900262, 1.3745507856490367);
        test_inverse_evaluate(q);
    }
    SECTION("Query 3")
    {
        Eigen::Matrix<Scalar, 1, 3> q(303.43838482686385, 231.21654572809825, 1.3745405013562231);
        test_inverse_evaluate(q);
    }
}
