#include <catch2/catch.hpp>
#include <iostream>

#include <nanospline/BezierPatch.h>
#include <nanospline/NURBSPatch.h>
#include <nanospline/OffsetPatch.h>
#include <nanospline/Sphere.h>

#include "validation_utils.h"

TEST_CASE("OffsetPatch", "[offset_patch][primitive]")
{
    using namespace nanospline;
    using Scalar = double;

    SECTION("Simple")
    {
        BezierPatch<Scalar, 3, 1, 1> base_patch;
        Eigen::Matrix<Scalar, 4, 3> control_grid;
        control_grid << 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0;
        base_patch.set_control_grid(control_grid);
        base_patch.initialize();

        OffsetPatch<Scalar, 3> patch;
        patch.set_base_surface(&base_patch);
        patch.set_offset(0.5);
        patch.initialize();

        validate_derivative(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10, 1e-6);
        validate_inverse_evaluation_3d(patch, 10, 10);
    }

    SECTION("Sphere")
    {
        Sphere<Scalar, 3> sphere;
        OffsetPatch<Scalar, 3> patch;
        patch.set_base_surface(&sphere);
        patch.set_offset(0.5);
        patch.initialize();
        validate_derivative(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10, 1e-6);
        validate_inverse_evaluation_3d(patch, 10, 10);
    }

    SECTION("Bezier patch 2")
    {
        BezierPatch<Scalar, 3, 3, 3> base_patch;
        Eigen::Matrix<Scalar, 16, 3> control_grid;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                control_grid.row(i * 4 + j) << j, i, ((i + j) % 2 == 0) ? -1 : 1;
            }
        }
        base_patch.set_control_grid(control_grid);
        base_patch.initialize();

        OffsetPatch<Scalar, 3> patch;
        patch.set_base_surface(&base_patch);
        patch.set_offset(0.5);
        patch.initialize();

        validate_derivative(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10, 1e-6);
        validate_inverse_evaluation_3d(patch, 10, 10);
    }

    SECTION("Simple NURBS patch")
    {
        NURBSPatch<Scalar, 3, 3, 3> base_patch;
        Eigen::Matrix<Scalar, 16, 3> control_grid;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                control_grid.row(i * 4 + j) << j, i, ((i + j) % 2 == 0) ? -1 : 1;
            }
        }
        base_patch.set_control_grid(control_grid);
        Eigen::Matrix<Scalar, 8, 1> knots_u, knots_v;
        knots_u << 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0;
        knots_v << 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 2.5, 2.5;
        base_patch.set_knots_u(knots_u);
        base_patch.set_knots_v(knots_v);

        Eigen::Matrix<Scalar, 16, 1> weights;
        weights.setOnes();
        base_patch.set_weights(weights);
        base_patch.initialize();

        OffsetPatch<Scalar, 3> patch;
        patch.set_base_surface(&base_patch);
        patch.set_offset(0.5);
        patch.set_u_lower_bound(base_patch.get_u_lower_bound());
        patch.set_u_upper_bound(base_patch.get_u_upper_bound());
        patch.set_v_lower_bound(base_patch.get_v_lower_bound());
        patch.set_v_upper_bound(base_patch.get_v_upper_bound());
        patch.initialize();

        // save_msh<Scalar>("debug.msh", {}, {&base_patch, &patch});

        validate_derivative(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10, 1e-6);
        validate_inverse_evaluation_3d(patch, 10, 10);
    }

    SECTION("NURBS patch")
    {
        NURBSPatch<Scalar, 3, -1, -1> base_patch;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 3> control_grid(14, 3);
        control_grid << 31.75, 0.0, 12.700000000000001, 31.75, 0.0, 0.0, 31.75, -54.99261314031184,
            12.700000000000001, 31.75, -54.99261314031184, 0.0, -15.874999999999993,
            -27.49630657015593, 12.700000000000001, -15.874999999999993, -27.49630657015593, 0.0,
            -63.499999999999986, -7.776507174585691e-15, 12.700000000000001, -63.499999999999986,
            -7.776507174585691e-15, 0.0, -15.875000000000014, 27.49630657015592, 12.700000000000001,
            -15.875000000000014, 27.49630657015592, 0.0, 31.74999999999995, 54.99261314031187,
            12.700000000000001, 31.74999999999995, 54.99261314031187, 0.0, 31.75,
            7.776507174585693e-15, 12.700000000000001, 31.75, 7.776507174585693e-15, 0.0;
        base_patch.set_control_grid(control_grid);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> weights(14, 1);
        weights << 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0;
        base_patch.set_weights(weights);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> u_knots(10, 1);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> v_knots(4, 1);
        u_knots << 0.0, 0.0, 0.0, 2.0943951023931953, 2.0943951023931953, 4.1887902047863905,
            4.1887902047863905, 6.283185307179586, 6.283185307179586, 6.283185307179586;
        v_knots << -10.573884999451131, -10.573884999451131, 2.12611500054887, 2.12611500054887;
        base_patch.set_knots_u(u_knots);
        base_patch.set_knots_v(v_knots);
        base_patch.set_degree_u(2);
        base_patch.set_degree_v(1);
        base_patch.set_periodic_u(true);
        base_patch.initialize();

        OffsetPatch<Scalar, 3> patch;
        patch.set_base_surface(&base_patch);
        patch.set_offset(1.5);
        patch.set_u_lower_bound(base_patch.get_u_lower_bound());
        patch.set_u_upper_bound(base_patch.get_u_upper_bound());
        patch.set_v_lower_bound(base_patch.get_v_lower_bound());
        patch.set_v_upper_bound(base_patch.get_v_upper_bound());
        patch.initialize();

        // save_msh<Scalar>("debug.msh", {}, {&base_patch, &patch});

        validate_derivative(patch, 10, 10, 1e-4);
        validate_inverse_evaluation(patch, 10, 10, 1e-6);
        validate_inverse_evaluation_3d(patch, 10, 10);
    }

    SECTION("Periodic NURBS patch")
    {
        NURBSPatch<Scalar, 3, -1, -1> base_patch;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 3> control_grid(28, 3);
        control_grid << 12.7, 6.35, 31.3416, 10.9942, 6.35, 31.3416, 10.9942, 6.35, 33.0474, 12.7,
            6.35, 33.0474, 14.4058, 6.35, 33.0474, 14.4058, 6.35, 31.3416, 12.7, 6.35, 31.3416,
            12.7, 6.35, 29.6707, 7.65245, 6.35, 29.6707, 7.65245, 6.35, 34.7183, 12.7, 6.35,
            34.7183, 17.7476, 6.35, 34.7183, 17.7476, 6.35, 29.6707, 12.7, 6.35, 29.6707, 12.7,
            5.38532, 28.3065, 4.92392, 5.38532, 28.3065, 4.92392, 5.38532, 36.0825, 12.7, 5.38532,
            36.0825, 20.4761, 5.38532, 36.0825, 20.4761, 5.38532, 28.3065, 12.7, 5.38532, 28.3065,
            12.7, 3.81, 27.7495, 3.81, 3.81, 27.7495, 3.81, 3.81, 36.6395, 12.7, 3.81, 36.6395,
            21.59, 3.81, 36.6395, 21.59, 3.81, 27.7495, 12.7, 3.81, 27.7495;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> knots_u(8);
        knots_u << 6.91638e-08, 6.91638e-08, 6.91638e-08, 6.91638e-08, 6.28319, 6.28319, 6.28319,
            6.28319;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> knots_v(11);
        knots_v << 0.955317, 0.955317, 0.955317, 0.955317, 1.5708, 1.5708, 1.5708, 2.18628, 2.18628,
            2.18628, 2.18628;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> weights(28);
        weights << 1, 0.333333, 0.333333, 1, 0.333333, 0.333333, 1, 0.877664, 0.292555, 0.292555,
            0.877664, 0.292555, 0.292555, 0.877664, 0.877664, 0.292555, 0.292555, 0.877664,
            0.292555, 0.292555, 0.877664, 1, 0.333333, 0.333333, 1, 0.333333, 0.333333, 1;

        base_patch.set_control_grid(control_grid);
        base_patch.set_knots_u(knots_u);
        base_patch.set_knots_v(knots_v);
        base_patch.set_weights(weights);
        base_patch.set_degree_u(3);
        base_patch.set_degree_v(3);
        base_patch.set_periodic_v(true);
        base_patch.initialize();

        OffsetPatch<Scalar, 3> patch;
        patch.set_base_surface(&base_patch);
        patch.set_offset(0.2);
        patch.set_u_lower_bound(0);
        patch.set_u_upper_bound(2 * M_PI);
        patch.set_v_lower_bound(0.96);
        patch.set_v_upper_bound(2.18);
        patch.initialize();

        // save_msh<Scalar>("debug.msh", {}, {&base_patch, &patch});

        SECTION("Query")
        {
            Eigen::Matrix<Scalar, 1, 3> q(
                14.843466516673436, 3.8766729726419129, 36.302470454378344);
            auto r = patch.inverse_evaluate(q,
                patch.get_u_lower_bound(),
                patch.get_u_upper_bound(),
                patch.get_v_lower_bound(),
                patch.get_v_upper_bound());
            const auto& uv = std::get<0>(r);
            auto p = patch.evaluate(uv[0], uv[1]);

            std::vector<Bezier<Scalar, 3, 1>> lines;
            lines.reserve(static_cast<size_t>(2));
            std::vector<const CurveBase<Scalar, 3>*> curves;
            auto add_line = [&](const auto& v0, const auto& v1) {
                Bezier<Scalar, 3, 1> line;
                Eigen::Matrix<Scalar, 2, 3> ctrl_pts;
                ctrl_pts.row(0) = v0;
                ctrl_pts.row(1) = v1;
                line.set_control_points(ctrl_pts);
                line.initialize();
                lines.push_back(std::move(line));
                curves.push_back(&lines.back());
            };
            add_line(q, p);

            save_msh<Scalar>("debug.msh", curves, {&patch});
            REQUIRE(std::get<1>(r));
        }

        validate_derivative(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10, 1e-6);
        validate_inverse_evaluation_3d(patch, 10, 10);
    }
}
