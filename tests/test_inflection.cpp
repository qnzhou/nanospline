#include <catch2/catch.hpp>
#include <iostream>

#include <nanospline/inflection.h>
#include <nanospline/split.h>
#include <nanospline/save_svg.h>

TEST_CASE("inflection", "[inflection]") {
    using namespace nanospline;
    using Scalar = double;

    SECTION("No inflection points") {
        Eigen::Matrix<Scalar, 4, 2, Eigen::RowMajor> control_pts;
        control_pts << 0.0, 0.0,
                       0.0, 1.0,
                       1.0, 1.0,
                       1.0, 0.0;

        Bezier<Scalar, 2, 3> curve;
        curve.set_control_points(control_pts);
        auto inflections = compute_inflections(curve);

        REQUIRE(inflections.size() == 0);
    }

    SECTION("Symmetric curve") {
        Eigen::Matrix<Scalar, 4, 2, Eigen::RowMajor> control_pts;
        control_pts << 0.0, 0.0,
                       0.0, 1.0,
                       1.0,-1.0,
                       1.0, 0.0;

        Bezier<Scalar, 2, 3> curve;
        curve.set_control_points(control_pts);
        auto inflections = compute_inflections(curve);

        REQUIRE(inflections.size() == 1);
        REQUIRE(inflections[0] == Approx(0.5));
    }

    SECTION("Assymetric curve") {
        Eigen::Matrix<Scalar, 4, 2, Eigen::RowMajor> control_pts;
        control_pts << 0.0, 0.0,
                       0.0, 1.0,
                       0.99,-1.0,
                       1.0, 0.0;

        Bezier<Scalar, 2, 3> curve;
        curve.set_control_points(control_pts);
        auto inflections = compute_inflections(curve);
        REQUIRE(inflections.size() == 1);
        REQUIRE(inflections[0] == Approx(0.49874999));
        REQUIRE(inflections[0] < 0.5);

        auto curvature = curve.evaluate_curvature(inflections[0]);
        REQUIRE(curvature.norm() == Approx(0.0).margin(1e-12));
    }

    SECTION("Almost collinear") {
        Eigen::Matrix<Scalar, 4, 2, Eigen::RowMajor> control_pts;
        control_pts << 0.0, 0.0,
                       0.0, 1.0,
                       0.0, 2.0,
                       1.0, 0.0;

        Bezier<Scalar, 2, 3> curve;
        curve.set_control_points(control_pts);
        auto inflections = compute_inflections(curve);
        REQUIRE(inflections.size() == 1);
        REQUIRE(inflections[0] == Approx(0.0));

        auto curvature = curve.evaluate_curvature(inflections[0]);
        REQUIRE(curvature.norm() == Approx(0.0).margin(1e-12));
    }

    SECTION("Completely collinear") {
        Eigen::Matrix<Scalar, 4, 2, Eigen::RowMajor> control_pts;
        control_pts << 0.0, 0.0,
                       0.0, 1.0,
                       0.0, 2.0,
                       0.0, 3.0;

        Bezier<Scalar, 2, 3> curve;
        curve.set_control_points(control_pts);
        REQUIRE_THROWS(compute_inflections(curve));
    }

    SECTION("Quadratic Bezier") {
        Eigen::Matrix<Scalar, 3, 2, Eigen::RowMajor> control_pts;
        control_pts << 0.0, 0.0,
                       1.0, 1.0,
                       0.0, 2.0;

        Bezier<Scalar, 2, 2> curve;
        curve.set_control_points(control_pts);
        auto inflections = compute_inflections(curve);
        REQUIRE(inflections.size() == 0);
    }

    SECTION("Quartic Bezier") {
        Eigen::Matrix<Scalar, 5, 2, Eigen::RowMajor> control_pts;
        control_pts << 0.0, 0.0,
                       0.0, 1.0,
                       1.0, 1.0,
                       1.0, 0.0,
                       2.0, 0.0;

        Bezier<Scalar, 2, 4> curve;
        curve.set_control_points(control_pts);
        auto inflections = compute_inflections(curve);
        REQUIRE(inflections.size() == 1);

        auto curvature = curve.evaluate_curvature(inflections[0]);
        REQUIRE(curvature.norm() == Approx(0.0).margin(1e-12));
    }

    SECTION("Quartic Bezier 2") {
        Eigen::Matrix<Scalar, 5, 2, Eigen::RowMajor> control_pts;
        control_pts << 0.0, 0.0,
                       1.0, 1.0,
                       2.0,-2.0,
                       3.0, 1.0,
                       4.0, 0.0;

        Bezier<Scalar, 2, 4> curve;
        curve.set_control_points(control_pts);
        auto inflections = compute_inflections(curve);
        REQUIRE(inflections.size() == 2);

        for (auto t : inflections) {
            auto curvature = curve.evaluate_curvature(t);
            REQUIRE(curvature.norm() == Approx(0.0).margin(1e-12));
        }
    }

}
