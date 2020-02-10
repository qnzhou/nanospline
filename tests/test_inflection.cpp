#include <catch2/catch.hpp>
#include <iostream>

#include <nanospline/Bezier.h>
#include <nanospline/RationalBezier.h>
#include <nanospline/split.h>
#include <nanospline/save_svg.h>

#include "forward_declaration.h"

TEST_CASE("inflection", "[inflection]") {
    using namespace nanospline;
    using Scalar = double;

    SECTION("No inflection points") {
        Eigen::Matrix<Scalar, 4, 2, Eigen::RowMajor> control_pts;
        control_pts << 0.0, 0.0,
                       0.0, 1.0,
                       1.0, 1.0,
                       1.0, 0.0;

        using Curve = Bezier<Scalar, 2, 3>;
        Curve curve;
        curve.set_control_points(control_pts);
        auto inflections = curve.compute_inflections(0.0, 1.0);

        REQUIRE(inflections.size() == 0);
    }

    SECTION("Symmetric curve") {
        Eigen::Matrix<Scalar, 4, 2, Eigen::RowMajor> control_pts;
        control_pts << 0.0, 0.0,
                       0.0, 1.0,
                       1.0,-1.0,
                       1.0, 0.0;

        using Curve = Bezier<Scalar, 2, 3>;
        Curve curve;
        curve.set_control_points(control_pts);
        auto inflections = curve.compute_inflections(0.0, 1.0);

        REQUIRE(inflections.size() == 1);
        REQUIRE(inflections[0] == Approx(0.5));
    }

    SECTION("Assymetric curve") {
        Eigen::Matrix<Scalar, 4, 2, Eigen::RowMajor> control_pts;
        control_pts << 0.0, 0.0,
                       0.0, 1.0,
                       0.99,-1.0,
                       1.0, 0.0;

        using Curve = Bezier<Scalar, 2, 3>;
        Curve curve;
        curve.set_control_points(control_pts);
        auto inflections = curve.compute_inflections(0.0, 1.0);
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

        using Curve = Bezier<Scalar, 2, 3>;
        Curve curve;
        curve.set_control_points(control_pts);
        auto inflections = curve.compute_inflections(0.0, 1.0);
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

        using Curve = Bezier<Scalar, 2, 3>;
        Curve curve;
        curve.set_control_points(control_pts);
        REQUIRE_THROWS(curve.compute_inflections(0.0, 1.0));
    }

    SECTION("Quadratic Bezier") {
        Eigen::Matrix<Scalar, 3, 2, Eigen::RowMajor> control_pts;
        control_pts << 0.0, 0.0,
                       1.0, 1.0,
                       0.0, 2.0;

        using Curve = Bezier<Scalar, 2, 2>;
        Curve curve;
        curve.set_control_points(control_pts);
        auto inflections = curve.compute_inflections(0.0, 1.0);
        REQUIRE(inflections.size() == 0);
    }

    SECTION("Quartic Bezier") {
        Eigen::Matrix<Scalar, 5, 2, Eigen::RowMajor> control_pts;
        control_pts << 0.0, 0.0,
                       0.0, 1.0,
                       1.0, 1.0,
                       1.0, 0.0,
                       2.0, 0.0;

        using Curve = Bezier<Scalar, 2, 4>;
        Curve curve;
        curve.set_control_points(control_pts);
        auto inflections = curve.compute_inflections(0.0, 1.0);
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

        using Curve = Bezier<Scalar, 2, 4>;
        Curve curve;
        curve.set_control_points(control_pts);
        auto inflections = curve.compute_inflections(0.0, 1.0);
        REQUIRE(inflections.size() == 2);

        for (auto t : inflections) {
            auto curvature = curve.evaluate_curvature(t);
            REQUIRE(curvature.norm() == Approx(0.0).margin(1e-12));
        }
    }

    SECTION("RationalBezier") {
        using Curve = RationalBezier<Scalar, 2, 3, true>;
        Curve curve;

        Eigen::Matrix<Scalar, 4, 2> control_pts;
        control_pts << 0.0, 0.0,
                       0.0, 1.0,
                       1.0,-1.0,
                       1.0, 0.0;
        curve.set_control_points(control_pts);

        Eigen::Matrix<Scalar, 4, 1> weights;
        int expected_num_inflections = 0;

        SECTION("Uniform weights") {
            weights.setConstant(1.1);
            expected_num_inflections = 1;
        }
        SECTION("Non-uniform weights") {
            weights << 1.0, 1.0, 0.0, 1.0;
            expected_num_inflections = 0;
        }
        SECTION("Large weights") {
            weights << 1.0, 1.0, 20.0, 1.0;
            expected_num_inflections = 1;
        }

        curve.set_weights(weights);
        curve.initialize();
        REQUIRE(curve.get_degree() == 3);
        auto inflections = curve.compute_inflections(1e-12, 1.0-1e-12);

        CHECK(inflections.size() == expected_num_inflections);
        for (auto t : inflections) {
            auto curvature = curve.evaluate_curvature(t);
            REQUIRE(curvature.norm() == Approx(0.0).margin(1e-12));
        }
    }

    SECTION("RationalBezier quartic") {
        using Curve = RationalBezier<Scalar, 2, 4, true>;
        Curve curve;

        Eigen::Matrix<Scalar, 5, 2> control_pts;
        control_pts << 0.0, 0.0,
                       1.0, 1.0,
                       2.0, 0.0,
                       3.0, 1.0,
                       4.0, 0.0;
        curve.set_control_points(control_pts);

        Eigen::Matrix<Scalar, 5, 1> weights;
        int expected_num_inflections = 0;

        SECTION("Uniform weights") {
            weights.setConstant(1.1);
            expected_num_inflections = 0;
        }
        SECTION("Non-uniform weights") {
            weights << 1.0, 10.0, 1.0, 10.0, 1.0;
            expected_num_inflections = 1;
        }

        curve.set_weights(weights);
        curve.initialize();
        REQUIRE(curve.get_degree() == 4);
        auto inflections = curve.compute_inflections(1e-12, 1.0-1e-12);

        CHECK(inflections.size() == expected_num_inflections);
        for (auto t : inflections) {
            auto curvature = curve.evaluate_curvature(t);
            REQUIRE(curvature.norm() == Approx(0.0).margin(1e-12));
        }
    }
}
