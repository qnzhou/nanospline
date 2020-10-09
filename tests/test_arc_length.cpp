#include <catch2/catch.hpp>

#include <vector>

#include <nanospline/BSpline.h>
#include <nanospline/Bezier.h>
#include <nanospline/BezierPatch.h>
#include <nanospline/NURBS.h>
#include <nanospline/arc_length.h>
#include <nanospline/forward_declaration.h>
#include <nanospline/save_obj.h>
#include "validation_utils.h"

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
            REQUIRE(t1 == Approx(t2));
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

        REQUIRE(diag_1 == Approx(diag_2));
        REQUIRE(diag_3 == Approx(diag_4));

        constexpr size_t N = 10;
        for (size_t i = 0; i <= N; i++) {
            Scalar t = (Scalar)i / (Scalar)N;

            Scalar u = u_min + t * (u_max - u_min);
            Scalar v = v_min + t * (v_max - v_min);

            auto l = arc_length(patch, u_min, v_min, u, v);
            auto p = inverse_arc_length(patch, u_min, v_min, u_max, v_max, l);
            REQUIRE(p[0] == Approx(u).margin(1e-6 * (u_max - u_min)));
            REQUIRE(p[1] == Approx(v).margin(1e-6 * (v_max - v_min)));
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
}
