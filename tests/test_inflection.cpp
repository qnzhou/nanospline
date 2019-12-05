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

        auto halves = split(curve, inflections[0]);

        auto curvature = curve.evaluate_curvature(inflections[0]);
        std::cout<<curvature.norm()<<std::endl;
        REQUIRE(curvature.norm() == Approx(0.0).margin(1e-12));
    }
}
