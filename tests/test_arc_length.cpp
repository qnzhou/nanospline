#include <nanospline/BSpline.h>
#include <nanospline/Bezier.h>
#include <nanospline/BezierPatch.h>
#include <nanospline/NURBS.h>
#include <nanospline/NURBSPatch.h>
#include <nanospline/arc_length.h>
#include <nanospline/forward_declaration.h>
#include <nanospline/save_obj.h>
#include "validation_utils.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <vector>

TEST_CASE("arc_length", "[arc_length]")
{
    using namespace nanospline;
    using Scalar = double;

    auto check_arc_length = [](auto& curve) {
        const auto t_min = curve.get_domain_lower_bound();
        const auto t_max = curve.get_domain_upper_bound();
        constexpr size_t N = 10;
        std::vector<Scalar> lengths(N + 1, 0.0);
        for (size_t i = 0; i <= N; i++) {
            Scalar t = (Scalar)i / (Scalar)N * (t_max - t_min) + t_min;
            lengths[i] = arc_length(curve, t);
        }

        for (size_t i = 0; i <= N; i++) {
            const Scalar t1 = (Scalar)i / (Scalar)N * (t_max - t_min) + t_min;
            const Scalar t2 = inverse_arc_length(curve, lengths[i]);
            REQUIRE_THAT(t1, Catch::Matchers::WithinAbs(t2, 1e-6));
        }
    };

    auto check_arc_length_on_patch = [](auto& patch) {
        const auto u_min = patch.get_u_lower_bound();
        const auto u_max = patch.get_u_upper_bound();
        const auto v_min = patch.get_v_lower_bound();
        const auto v_max = patch.get_v_upper_bound();

        const auto diag_1 = arc_length(patch, u_min, v_min, u_max, v_max);
        const auto diag_2 = arc_length(patch, u_max, v_max, u_min, v_min);
        const auto diag_3 = arc_length(patch, u_min, v_max, u_max, v_min);
        const auto diag_4 = arc_length(patch, u_max, v_min, u_min, v_max);

        REQUIRE_THAT(diag_1, Catch::Matchers::WithinAbs(diag_2, 1e-6));
        REQUIRE_THAT(diag_3, Catch::Matchers::WithinAbs(diag_4, 1e-6));

        constexpr size_t N = 10;
        for (size_t i = 0; i <= N; i++) {
            Scalar t = (Scalar)i / (Scalar)N;

            Scalar u = u_min + t * (u_max - u_min);
            Scalar v = v_min + t * (v_max - v_min);

            auto l = arc_length(patch, u_min, v_min, u, v);
            auto p = inverse_arc_length(patch, u_min, v_min, u_max, v_max, l);
            REQUIRE_THAT(p[0], Catch::Matchers::WithinAbs(u, 1e-6 * (u_max - u_min)));
            REQUIRE_THAT(p[1], Catch::Matchers::WithinAbs(v, 1e-6 * (v_max - v_min)));
        }
    };

    SECTION("Bezier degree 3")
    {
        Eigen::Matrix<Scalar, 4, 2> ctrl_pts;
        ctrl_pts << 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0;
        Bezier<Scalar, 2, 3> curve;
        curve.set_control_points(ctrl_pts);

        check_arc_length(curve);
    }

    SECTION("BSpline degree 3")
    {
        Eigen::Matrix<Scalar, 10, 2> ctrl_pts;
        ctrl_pts << 1, 4, .5, 6, 5, 4, 3, 12, 11, 14, 8, 4, 12, 3, 11, 9, 15, 10, 17, 8;
        Eigen::Matrix<Scalar, 14, 1> knots;
        knots << 0, 0, 0, 0, 1.0 / 7, 2.0 / 7, 3.0 / 7, 4.0 / 7, 5.0 / 7, 6.0 / 7, 1, 1, 1, 1;

        BSpline<Scalar, 2, 3, true> curve;
        curve.set_control_points(ctrl_pts);
        curve.set_knots(knots);

        check_arc_length(curve);
    }

    SECTION("Quarter circle")
    {
        constexpr Scalar R = 12.1;
        Eigen::Matrix<Scalar, 1, 2> c(0.0, R);
        Eigen::Matrix<Scalar, 9, 2> control_pts;
        control_pts << 0.0, 0.0, R, 0.0, R, R, R, 2 * R, 0.0, 2 * R, -R, 2 * R, -R, R, -R, 0.0, 0.0,
            0.0;
        Eigen::Matrix<Scalar, 12, 1> knots;
        knots << 0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0;
        Eigen::Matrix<Scalar, 9, 1> weights;
        weights << 1.0, sqrt(2) / 2, 1.0, sqrt(2) / 2, 1.0, sqrt(2) / 2, 1.0, sqrt(2) / 2, 1.0;

        NURBS<Scalar, 2, 2, true> curve;
        curve.set_control_points(control_pts);
        curve.set_knots(knots);
        curve.set_weights(weights);
        curve.initialize();

        check_arc_length(curve);
    }

    SECTION("Bilinear patch")
    {
        BezierPatch<Scalar, 3, 1, 1> patch;
        Eigen::Matrix<Scalar, 4, 3> control_grid;
        control_grid << 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0;
        patch.set_control_grid(control_grid);
        patch.initialize();

        check_arc_length_on_patch(patch);
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

        check_arc_length_on_patch(patch);
    }

    SECTION("NURBS patch")
    {
        NURBSPatch<Scalar, 3, -1, -1> patch;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 3> control_grid(6, 3);
        control_grid.row(0) << 6.455076766863815, 3.6966766140380294, 0.0;
        control_grid.row(1) << 6.455076766863815, 3.6966766140380294, 94.60000000000001;
        control_grid.row(2) << -2.3528094251012197e-13, 6.1899665468625535, 0.0;
        control_grid.row(3) << -2.3528094251012197e-13, 6.1899665468625535, 94.60000000000001;
        control_grid.row(4) << -6.4550767668642415, 3.6966766140378624, 0.0;
        control_grid.row(5) << -6.4550767668642415, 3.6966766140378624, 94.60000000000001;
        patch.set_control_grid(control_grid);

        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> knots_u(6), knots_v(4);
        knots_u << 4.343789729359035, 4.343789729359035, 4.343789729359035, 5.0809882314103705,
            5.0809882314103705, 5.0809882314103705;
        knots_v << -94.60000000000001, -94.60000000000001, 0.0, 0.0;
        patch.set_knots_u(knots_u);
        patch.set_knots_v(knots_v);

        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> weights(6);
        weights << 1.0, 1.0, 0.932832963227302, 0.932832963227302, 1.0, 1.0;
        patch.set_weights(weights);
        patch.set_degree_u(2);
        patch.set_degree_v(1);
        patch.initialize();

        check_arc_length_on_patch(patch);
    }

    SECTION("Periodic curve")
    {
        constexpr Scalar R = 1;
        Eigen::Matrix<Scalar, 1, 2> c(0.0, R);
        Eigen::Matrix<Scalar, 9, 2> control_pts;
        control_pts << 0.0, 0.0, R, 0.0, R, R, R, 2 * R, 0.0, 2 * R, -R, 2 * R, -R, R, -R, 0.0, 0.0,
            0.0;
        Eigen::Matrix<Scalar, 12, 1> knots;
        knots << 0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0;
        Eigen::Matrix<Scalar, 9, 1> weights;
        weights << 1.0, sqrt(2) / 2, 1.0, sqrt(2) / 2, 1.0, sqrt(2) / 2, 1.0, sqrt(2) / 2, 1.0;

        NURBS<Scalar, 2, 2, true> curve;
        curve.set_control_points(control_pts);
        curve.set_knots(knots);
        curve.set_weights(weights);
        curve.set_periodic(true);
        curve.initialize();

        const Scalar t_min = curve.get_domain_lower_bound();
        const Scalar t_max = curve.get_domain_upper_bound();
        auto single_round = arc_length(curve, t_min, t_max, 20);
        REQUIRE_THAT(single_round, Catch::Matchers::WithinRel(2 * M_PI * R, 1e-3));
        auto double_round = arc_length(curve, t_min, t_max * 2 - t_min, 20);
        REQUIRE_THAT(double_round, Catch::Matchers::WithinRel(4 * M_PI * R, 1e-3));
    }

    SECTION("Perioidc patch")
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

        auto single_round = arc_length(patch, u_min, v_mid, u_max, v_mid);
        auto double_round = arc_length(patch, u_min, v_mid, 2 * u_max - u_min, v_mid);
        REQUIRE_THAT(double_round, Catch::Matchers::WithinAbs(single_round * 2, 1e-3));
        auto double_spiral = arc_length(patch, u_min, v_min, 2 * u_max - u_min, v_max);
        REQUIRE(double_spiral > double_round);
    }
}
