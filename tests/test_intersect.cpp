#include <catch2/catch.hpp>

#include <nanospline/forward_declaration.h>
#include <nanospline/nanospline.h>
#include "validation_utils.h"

TEST_CASE("Curve-plane intersect", "[intersect]")
{
    using namespace nanospline;
    using Scalar = double;

    Scalar t;
    bool converged;

    SECTION("Line-plane")
    {
        Line<Scalar, 3> line;

        SECTION("case 1")
        {
            line.set_location({0, 0, 0});
            line.set_direction({1, 1, 1});
        }
        SECTION("case 2")
        {
            line.set_location({0.5, 0, 0});
            line.set_direction({1, 1, 1});
        }
        SECTION("case 3")
        {
            line.set_location({5, 5, 5});
            line.set_direction({1, 1, 1});
        }

        line.initialize();
        std::array<Scalar, 4> plane{1, 1, 1, -1};

        std::tie(t, converged) = intersect(line, plane, 0., 10, 1e-3);
        REQUIRE(converged);
        auto p = line.evaluate(t);

        REQUIRE(p.sum() == Approx(1).margin(1e-3));
    }

    SECTION("Parallel")
    {
        Line<Scalar, 3> line;
        line.set_location({1, 1, 1});
        line.set_direction({1, 0, 0});
        line.initialize();

        std::array<Scalar, 4> plane{0, 1, 0, 0};
        std::tie(t, converged) = intersect(line, plane, 0., 10, 1e-3);
        REQUIRE_FALSE(converged);
    }

    SECTION("Circle-plane")
    {
        Circle<Scalar, 3> circle;
        circle.set_center({0, 0, 0});
        circle.set_radius(1);
        circle.initialize();
        std::array<Scalar, 4> plane{1, 1, 1, -1};

        Scalar t0, t1, t2, t3;
        std::tie(t0, converged) = intersect(circle, plane, 0., 10, 1e-3);
        REQUIRE(converged);
        std::tie(t1, converged) = intersect(circle, plane, M_PI / 2, 10, 1e-3);
        REQUIRE(converged);
        std::tie(t2, converged) = intersect(circle, plane, 0.1, 10, 1e-3);
        REQUIRE(converged);
        std::tie(t3, converged) = intersect(circle, plane, 1.5, 10, 1e-3);
        REQUIRE(converged);

        REQUIRE(t0 == Approx(0).margin(1e-3));
        REQUIRE(t1 == Approx(M_PI / 2).margin(1e-3));
        REQUIRE(t2 == Approx(0).margin(1e-3));
        REQUIRE(t3 == Approx(M_PI / 2).margin(1e-3));

        SECTION("No intersections")
        {
            std::tie(t, converged) = intersect(circle, {1, 1, 1, 2}, 0., 10, 1e-3);
            REQUIRE_FALSE(converged);
        }
    }

    SECTION("Degenerate curve")
    {
        Circle<Scalar, 3> circle;
        circle.set_center({1, 0, 0});
        circle.set_radius(0);
        circle.initialize();

        SECTION("Do intersect")
        {
            std::array<Scalar, 4> plane{1, 1, 1, -1};
            std::tie(t, converged) = intersect(circle, plane, 0., 10, 1e-3);
            REQUIRE(converged);
        }

        SECTION("Do not intersect")
        {
            std::array<Scalar, 4> plane{1, 1, 1, -2};
            std::tie(t, converged) = intersect(circle, plane, 0., 10, 1e-3);
            REQUIRE_FALSE(converged);
        }
    }

    SECTION("Tangent")
    {
        Circle<Scalar, 3> circle;
        circle.set_center({0, 0, 0});
        circle.set_radius(1);
        circle.initialize();
        std::array<Scalar, 4> plane{1, 0, 0, -1};

        SECTION("Initial guess is the solution")
        {
            std::tie(t, converged) = intersect(circle, plane, 0., 10, 1e-3);
            REQUIRE(converged);
            REQUIRE(t == Approx(0).margin(1e-3));
        }
        SECTION("Initial guess is near the solution")
        {
            // Because the tangency, the tolerance needs to be smaller to get
            // an accurate solution.
            std::tie(t, converged) = intersect(circle, plane, 0.1, 10, 1e-6);
            REQUIRE(converged);
            REQUIRE(t == Approx(0).margin(1e-3));
        }
        SECTION("Initial guess is far from the solution")
        {
            // Need larger number of iteration and crazy small tolerance to
            // work.
            std::tie(t, converged) = intersect(circle, plane, M_PI / 2, 100, 1e-12);
            REQUIRE(converged);
            REQUIRE(t == Approx(0).margin(1e-3));
        }
    }

    SECTION("Nearly Tangent but not touching")
    {
        Circle<Scalar, 3> circle;
        circle.set_center({0, 0, 0});
        circle.set_radius(1);
        circle.initialize();
        std::array<Scalar, 4> plane{1, 0, 0, -1.001};

        std::tie(t, converged) = intersect(circle, plane, 0., 10, 1e-6);
        REQUIRE_FALSE(converged);
        std::tie(t, converged) = intersect(circle, plane, 0.1, 10, 1e-6);
        REQUIRE_FALSE(converged);
    }

    SECTION("Generic 3D curve")
    {
        Eigen::Matrix<Scalar, 4, 3> control_pts;
        control_pts << 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0;
        Bezier<Scalar, 3> curve;
        curve.set_control_points(control_pts);

        SECTION("Single intersection")
        {
            std::array<Scalar, 4> plane{1, 1, 0, -1};

            std::tie(t, converged) = intersect(curve, plane, 0., 10, 1e-3);
            REQUIRE(converged);
            REQUIRE(t == Approx(0.5).margin(1e-3));

            std::tie(t, converged) = intersect(curve, plane, 1., 10, 1e-3);
            REQUIRE(converged);
            REQUIRE(t == Approx(0.5).margin(1e-3));

            std::tie(t, converged) = intersect(curve, plane, 0.2, 10, 1e-3);
            REQUIRE(converged);
            REQUIRE(t == Approx(0.5).margin(1e-3));

            std::tie(t, converged) = intersect(curve, plane, 0.9, 10, 1e-3);
            REQUIRE(converged);
            REQUIRE(t == Approx(0.5).margin(1e-3));
        }

        SECTION("Multiple intersections")
        {
            std::array<Scalar, 4> plane{-2, 1, 0, 0.5};
            auto validate = [&](Scalar t) {
                auto p = curve.evaluate(t);
                REQUIRE(p[0] * plane[0] + p[1] * plane[1] + p[2] * plane[2] + plane[3] ==
                        Approx(0).margin(1e-3));
            };

            Scalar t1, t2, t3, t4, t5, t6;

            std::tie(t1, converged) = intersect(curve, plane, 0., 10, 1e-6);
            REQUIRE(converged);
            std::tie(t2, converged) = intersect(curve, plane, 0.1, 10, 1e-6);
            REQUIRE(converged);
            REQUIRE(t1 == Approx(t2).margin(1e-3));
            validate(t1);
            validate(t2);

            std::tie(t3, converged) = intersect(curve, plane, 0.4, 10, 1e-6);
            REQUIRE(converged);
            std::tie(t4, converged) = intersect(curve, plane, 0.6, 10, 1e-6);
            REQUIRE(converged);
            REQUIRE(t3 == Approx(t4).margin(1e-3));
            REQUIRE(t3 == Approx(0.5).margin(1e-3));
            validate(t3);
            validate(t4);

            std::tie(t5, converged) = intersect(curve, plane, 0.9, 10, 1e-6);
            REQUIRE(converged);
            std::tie(t6, converged) = intersect(curve, plane, 1.0, 10, 1e-6);
            REQUIRE(converged);
            REQUIRE(t5 == Approx(t6).margin(1e-3));
            validate(t5);
            validate(t6);
        }
    }
}

TEST_CASE("Curve-in-patch-plane intersect", "[intersect]")
{
    using namespace nanospline;
    using Scalar = double;

    SECTION("Line plane")
    {
        Plane<Scalar, 3> patch;
        patch.initialize();

        std::array<Scalar, 4> plane{1, 1, 0, -1};

        Scalar u, v;
        bool converged;
        std::tie(u, v, converged) = intersect(patch, 0., 0., 1., 1., plane, 0., 0., 10, 1e-3);
        REQUIRE(converged);
        REQUIRE(u == Approx(0.5).margin(1e-3));
        REQUIRE(v == Approx(0.5).margin(1e-3));

        std::tie(u, v, converged) = intersect(patch, 0., 0., 1., 0., plane, 0., 0., 10, 1e-3);
        REQUIRE(converged);
        REQUIRE(u == Approx(1).margin(1e-3));
        REQUIRE(v == Approx(0).margin(1e-3));

        std::tie(u, v, converged) = intersect(patch, 0., 0.1, 1., 0.1, plane, 0., 0., 10, 1e-3);
        REQUIRE(converged);
        REQUIRE(u == Approx(0.9).margin(1e-3));
        REQUIRE(v == Approx(0.1).margin(1e-3));
    }

    SECTION("curve on cylinder")
    {
        Cylinder<Scalar, 3> patch;
        Cylinder<Scalar, 3>::Frame frame;
        frame << 0, 1, 0, 0, 0, 1, 1, 0, 0;
        patch.set_frame(frame);
        patch.initialize();

        std::array<Scalar, 4> plane{1, 1, 0, -1};

        auto validate = [&](auto u, auto v) {
            auto p = patch.evaluate(u, v);
            REQUIRE(p[0] * plane[0] + p[1] * plane[1] + p[2] * plane[2] + plane[3] ==
                    Approx(0).margin(1e-3));
        };

        Scalar u, v;
        bool converged;

        SECTION("Straight line")
        {
            std::tie(u, v, converged) = intersect(patch, 0., 0., 0., 1., plane, 0., 0., 10, 1e-3);
            REQUIRE(converged);
            REQUIRE(u == Approx(0).margin(1e-3));
            REQUIRE(v == Approx(0).margin(1e-3));
            validate(u, v);
        }

        SECTION("Sprial")
        {
            std::tie(u, v, converged) =
                intersect(patch, M_PI, 0., 0., 1., plane, M_PI / 2, 0.5, 10, 1e-3);
            REQUIRE(converged);
            validate(u, v);
        }

        SECTION("Arc")
        {
            std::tie(u, v, converged) =
                intersect(patch, M_PI, 0.5, 0., 0.5, plane, M_PI / 2, 0.5, 10, 1e-3);
            REQUIRE(converged);
            validate(u, v);
        }

        SECTION("Tangent")
        {
            plane = {0, 1, 0, -1};
            std::tie(u, v, converged) =
                intersect(patch, M_PI, 0.5, 0., 0.5, plane, M_PI / 2, 0.5, 10, 1e-3);
            REQUIRE(converged);
            validate(u, v);
        }

        SECTION("Not intersecting")
        {
            plane = {0, 1, 0, -1.1};
            std::tie(u, v, converged) =
                intersect(patch, M_PI, 0.5, 0., 0.5, plane, M_PI / 2, 0.5, 10, 1e-3);
            REQUIRE_FALSE(converged);
        }

        SECTION("Multple intersections")
        {
            plane = {0, 1, 0, -0.5};

            std::tie(u, v, converged) =
                intersect(patch, M_PI, 0., -M_PI, 1., plane, M_PI, 0.3, 20, 1e-3);
            REQUIRE(converged);
            validate(u, v);

            Scalar u2, v2;
            std::tie(u2, v2, converged) =
                intersect(patch, M_PI, 0., -M_PI, 1., plane, -M_PI, 0.7, 20, 1e-3);
            REQUIRE(converged);
            validate(u2, v2);

            REQUIRE(std::hypot(u - u2, v - v2) > 0.1);
        }

        SECTION("Degenerate line")
        {
            std::tie(u, v, converged) =
                intersect(patch, 0., 0., 0., 0., plane, M_PI, 0.3, 20, 1e-3);
            REQUIRE(converged);
            validate(u, v);

            std::tie(u, v, converged) =
                intersect(patch, 0., 1., 0., 1., plane, M_PI, 0.3, 20, 1e-3);
            REQUIRE_FALSE(converged);
        }
    }
}

TEST_CASE("Generic curve-in-patch-plane intersect", "[intersect]")
{
    using namespace nanospline;
    using Scalar = double;

    auto validate =
        [](const auto& curve, const auto& patch, Scalar t, Scalar u, Scalar v, Scalar tol) {
            auto p0 = curve.evaluate(t);
            auto p1 = patch.evaluate(u, v);
            REQUIRE((p0 - p1).norm() == Approx(0).margin(tol));
        };

    SECTION("Line plane")
    {
        Line<Scalar, 3> line;
        line.set_location({0, 0, 0});
        line.set_direction({1, 1, 1});
        line.initialize();

        Plane<Scalar, 3> plane;
        Eigen::Matrix<Scalar, 2, 3> frame;
        frame << 1, 0, 0, 0, 1, 0;
        plane.set_location({0.0, 0.0, 0.5});
        plane.set_frame(frame);
        plane.initialize();

        Scalar t, u, v;
        bool converged;
        std::tie(t, u, v, converged) = intersect(line, plane, 0., 0., 0., 10, 1e-3);
        REQUIRE(converged);
        validate(line, plane, t, u, v, 1e-3);
    }

    SECTION("Parallel line and plane")
    {
        Line<Scalar, 3> line;
        line.set_location({0, 0, 0});
        line.set_direction({1, 0, 0});
        line.initialize();

        Plane<Scalar, 3> plane;
        Eigen::Matrix<Scalar, 2, 3> frame;
        frame << 1, 0, 0, 0, 1, 0;
        plane.set_location({0.0, 0.0, 0.5});
        plane.set_frame(frame);
        plane.initialize();

        Scalar t, u, v;
        bool converged;
        std::tie(t, u, v, converged) = intersect(line, plane, 0., 0., 0., 10, 1e-3);
        REQUIRE_FALSE(converged);
    }

    SECTION("Line Sphere")
    {
        Line<Scalar, 3> line;
        line.set_location({0, 0, 0});
        line.set_direction({1, 1, 1});
        line.initialize();

        Sphere<Scalar, 3> sphere;
        sphere.set_location({0, 0, 0});
        sphere.set_radius(1);
        sphere.initialize();

        Scalar t, u, v;
        bool converged;
        std::tie(t, u, v, converged) = intersect(line, sphere, 0.5, 0.5, 0.5, 10, 1e-3);
        REQUIRE(converged);
        validate(line, sphere, t, u, v, 1e-3);

        std::tie(t, u, v, converged) = intersect(line, sphere, -0.5, M_PI + 0.5, -0.5, 10, 1e-3);
        REQUIRE(converged);
        validate(line, sphere, t, u, v, 1e-3);
    }

    SECTION("Curve plane")
    {
        Bezier<Scalar, 3> curve;
        Eigen::Matrix<Scalar, 4, 3> control_points;
        control_points << 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1;
        curve.set_control_points(control_points);
        curve.initialize();

        Plane<Scalar, 3> plane;
        plane.set_location({0.5, 0.5, 0.5});
        Eigen::Matrix<Scalar, 2, 3> frame;
        frame << 1, 0, 0, 0, 1, 0;
        plane.set_frame(frame);
        plane.initialize();

        Scalar t, u, v;
        bool converged;
        std::tie(t, u, v, converged) = intersect(curve, plane, 0., 0., 0., 10, 1e-3);
        validate(curve, plane, t, u, v, 1e-3);
    }

    SECTION("Curve patch")
    {
        Bezier<Scalar, 3> curve;
        Eigen::Matrix<Scalar, 4, 3> control_points;
        control_points << 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1;
        curve.set_control_points(control_points);
        curve.initialize();

        BezierPatch<Scalar, 3, 3, 3> patch;
        Eigen::Matrix<Scalar, 16, 3> control_grid;
        control_grid << 0, 0, 1, 1, 0, 0, 2, 0, 1, 3, 0, 0, 0, 1, 0, 1, 1, 1, 2, 1, 0, 3, 1, 1, 0,
            2, 1, 1, 2, 0, 2, 2, 1, 3, 2, 0, 0, 3, 0, 1, 3, 1, 2, 3, 0, 3, 3, 1;
        patch.set_control_grid(control_grid);

        Scalar t, u, v;
        bool converged;
        std::tie(t, u, v, converged) = intersect(curve, patch, 0., 0., 0., 10, 1e-3);
        REQUIRE(converged);
        validate(curve, patch, t, u, v, 1e-3);

        std::tie(t, u, v, converged) = intersect(curve, patch, 0.9, 0.5, 0.5, 10, 1e-3);
        REQUIRE(converged);
        validate(curve, patch, t, u, v, 1e-3);
    }
}
