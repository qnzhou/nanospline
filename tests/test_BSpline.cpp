#include <catch2/catch.hpp>

#include <nanospline/BSpline.h>
#include <nanospline/Bezier.h>

TEST_CASE("BSpline", "[bspline]") {
    using namespace nanospline;

    SECTION("Generic degree 0") {
        Eigen::Matrix<float, 3, 2> control_pts;
        control_pts << 0.0, 0.0,
                       1.0, 0.0,
                       2.0, 0.0;
        Eigen::Matrix<float, 4, 1> knots;
        knots << 0.0, 0.5, 0.75, 1.0;

        BSpline<float, 2, 0, true> curve;
        curve.set_control_points(control_pts);
        curve.set_knots(knots);

        REQUIRE(curve.get_degree() == 0);

        auto p0 = curve.evaluate(0.1);
        REQUIRE(p0[0] == Approx(0.0));
        REQUIRE(p0[1] == Approx(0.0));

        auto p1 = curve.evaluate(0.6);
        REQUIRE(p1[0] == Approx(1.0));
        REQUIRE(p1[1] == Approx(0.0));

        auto p2 = curve.evaluate(1.0);
        REQUIRE(p2[0] == Approx(2.0));
        REQUIRE(p2[1] == Approx(0.0));
    }

    SECTION("Generic degree 1") {
        Eigen::Matrix<float, 4, 2> control_pts;
        control_pts << 0.0, 0.0,
                       1.0, 0.0,
                       2.0, 0.0,
                       3.0, 0.0;
        Eigen::Matrix<float, 6, 1> knots;
        knots << 0.0, 0.0, 0.2, 0.8, 1.0, 1.0;

        BSpline<float, 2, 1, true> curve;
        curve.set_control_points(control_pts);
        curve.set_knots(knots);

        auto p0 = curve.evaluate(0.1);
        REQUIRE(p0[0] == Approx(0.5));
        REQUIRE(p0[1] == Approx(0.0));

        auto p1 = curve.evaluate(0.5);
        REQUIRE(p1[0] == Approx(1.5));
        REQUIRE(p1[1] == Approx(0.0));

        auto p2 = curve.evaluate(0.9);
        REQUIRE(p2[0] == Approx(2.5));
        REQUIRE(p2[1] == Approx(0.0));
    }

    SECTION("Generic degree 2") {
        Eigen::Matrix<float, 4, 2> control_pts;
        control_pts << 0.0, 0.0,
                       1.0, 0.0,
                       2.0, 0.0,
                       3.0, 0.0;
        Eigen::Matrix<float, 7, 1> knots;
        knots << 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0;

        BSpline<float, 2, 2, true> curve;
        curve.set_control_points(control_pts);
        curve.set_knots(knots);

        auto p0 = curve.evaluate(0.0);
        REQUIRE(p0[0] == Approx(0.0));
        REQUIRE(p0[1] == Approx(0.0));

        auto p1 = curve.evaluate(0.5);
        REQUIRE(p1[0] == Approx(1.5));
        REQUIRE(p1[1] == Approx(0.0));

        auto p2 = curve.evaluate(1.0);
        REQUIRE(p2[0] == Approx(3.0));
        REQUIRE(p2[1] == Approx(0.0));

        auto p3 = curve.evaluate(0.1);
        auto p4 = curve.evaluate(0.9);
        REQUIRE(p3[0] == Approx(3.0 - p4[0]));
        REQUIRE(p3[1] == Approx(0.0));
        REQUIRE(p4[1] == Approx(0.0));
    }

    SECTION("Generic degree 3") {
        Eigen::Matrix<float, 4, 2> control_pts;
        control_pts << 0.0, 0.0,
                       1.0, 0.0,
                       2.0, 0.0,
                       3.0, 0.0;
        Eigen::Matrix<float, 8, 1> knots;
        knots << 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0;

        BSpline<float, 2, 3, true> curve;
        curve.set_control_points(control_pts);
        curve.set_knots(knots);

        // B-spline without internal knots is a Bezier curve.
        Bezier<float, 2, 3> bezier_curve;
        bezier_curve.set_control_points(control_pts);

        for (float t=0.0; t<1.01; t+=0.2) {
            auto p0 = curve.evaluate(t);
            auto p1 = bezier_curve.evaluate(t);
            REQUIRE(p0[0] == Approx(p1[0]));
            REQUIRE(p0[1] == Approx(p1[1]));
        }
    }
}
