#include <catch2/catch.hpp>
#include <iostream>

#include <nanospline/NURBSPatch.h>
#include <nanospline/forward_declaration.h>
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
        const auto uv0 = patch.inverse_evaluate(q, u_min, u_max, v_min, v_max);
        REQUIRE((patch.evaluate(uv0[0], uv0[1]) - q).norm() == Approx(0));
        const auto uv1 = patch.inverse_evaluate(q, u_max - d, u_max + d, v_min, v_max);
        REQUIRE((patch.evaluate(uv1[0], uv1[1]) - q).norm() == Approx(0));
        const auto uv2 =
            patch.inverse_evaluate(q, u_min - 2 * period - d, u_min - 2 * period + d, v_min, v_max);
        REQUIRE((patch.evaluate(uv2[0], uv2[1]) - q).norm() == Approx(0));
        const auto uv3 =
            patch.inverse_evaluate(q, u_min - period + d, u_max - period - d, v_min, v_max);
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
                patch.inverse_evaluate(p, u_min, u_max, v_min, v_max);
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

