#include <catch2/catch.hpp>

#include <nanospline/split.h>
#include <nanospline/save_svg.h>
#include "validation_utils.h"

TEST_CASE("split", "[split]") {
    using namespace nanospline;
    using Scalar = double;

    SECTION("Bezier degree 1") {
        Eigen::Matrix<Scalar, 2, 2> control_pts;
        control_pts << 0.0, 0.0,
                       1.0, 0.0;
        Bezier<Scalar, 2, 1, true> curve;
        curve.set_control_points(control_pts);

        const auto parts = split(curve, 0.5);
        REQUIRE(parts.size() == 2);

        const auto& ctrl_pts_1 = parts[0].get_control_points();
        const auto& ctrl_pts_2 = parts[1].get_control_points();
        REQUIRE((ctrl_pts_1.row(0) - control_pts.row(0)).norm() == Approx(0.0));
        REQUIRE((ctrl_pts_2.row(1) - control_pts.row(1)).norm() == Approx(0.0));
        REQUIRE((ctrl_pts_1.row(1) - ctrl_pts_2.row(0)).norm() == Approx(0.0));
        REQUIRE(ctrl_pts_1(1, 0) == Approx(0.5));
        REQUIRE(ctrl_pts_1(1, 1) == Approx(0.0));
    }

    SECTION("Bezier degree 3") {
        Eigen::Matrix<Scalar, 4, 2> control_pts;
        control_pts << 0.0, 0.0,
                       1.0, 1.0,
                       2.0, 1.0,
                       3.0, 0.0;
        Bezier<Scalar, 2, 3, true> curve;
        curve.set_control_points(control_pts);

        Scalar split_location = 0.0;
        SECTION("Beginning") {
            split_location = 0.0;
        }
        SECTION("Middle") {
            split_location = 0.5;
        }
        SECTION("2/3") {
            split_location = 2.0/3.0;
        }
        SECTION("End") {
            split_location = 1.0;
        }

        const auto parts = split(curve, split_location);
        assert_same(curve, parts[0], 10, 0.0, split_location, 0.0, 1.0);
        assert_same(curve, parts[1], 10, split_location, 1.0, 0.0, 1.0);
    }

    SECTION("Rational bezier degree 1") {
        Eigen::Matrix<Scalar, 2, 2> control_pts;
        control_pts << 0.0, 0.0,
                       1.0, 0.0;
        Eigen::Matrix<Scalar, 2, 1> weights;
        weights << 0.1, 1.0;

        RationalBezier<Scalar, 2, 1, true> curve;
        curve.set_control_points(control_pts);
        curve.set_weights(weights);
        curve.initialize();

        Scalar split_location = 0.0;
        SECTION("Beginning") {
            split_location = 0.0;
        }
        SECTION("Middle") {
            split_location = 0.5;
        }
        SECTION("2/3") {
            split_location = 2.0/3.0;
        }
        SECTION("End") {
            split_location = 1.0;
        }

        const auto parts = split(curve, split_location);
        assert_same(curve, parts[0], 10, 0.0, split_location, 0.0, 1.0);
        assert_same(curve, parts[1], 10, split_location, 1.0, 0.0, 1.0);
    }

    SECTION("Rational bezier degree 3") {
        Eigen::Matrix<Scalar, 4, 2> control_pts;
        control_pts << 0.0, 0.0,
                       1.0, 1.0,
                       2.0,-1.0,
                       3.0, 0.0;
        Eigen::Matrix<Scalar, 4, 1> weights;
        weights << 0.1, 1.0, 0.1, 0.1;

        RationalBezier<Scalar, 2, 3, true> curve;
        curve.set_control_points(control_pts);
        curve.set_weights(weights);
        curve.initialize();

        Scalar split_location = 0.0;
        SECTION("Beginning") {
            split_location = 0.0;
        }
        SECTION("Middle") {
            split_location = 0.5;
        }
        SECTION("2/3") {
            split_location = 2.0/3.0;
        }
        SECTION("End") {
            split_location = 1.0;
        }

        const auto parts = split(curve, split_location);
        assert_same(curve, parts[0], 10, 0.0, split_location, 0.0, 1.0);
        assert_same(curve, parts[1], 10, split_location, 1.0, 0.0, 1.0);
    }

    SECTION("BSpline degree 3") {
        Eigen::Matrix<Scalar, 10, 2> ctrl_pts;
        ctrl_pts << 1, 4, .5, 6, 5, 4, 3, 12, 11, 14, 8, 4, 12, 3, 11, 9, 15, 10, 17, 8;
        Eigen::Matrix<Scalar, 14, 1> knots;
        knots << 0, 0, 0, 0, 1.0/7, 2.0/7, 3.0/7, 4.0/7, 5.0/7, 6.0/7, 1, 1, 1, 1;

        BSpline<Scalar, 2, 3, true> curve;
        curve.set_control_points(ctrl_pts);
        curve.set_knots(knots);

        Scalar split_location = 0.0;
        SECTION("Beginning") {
            split_location = 0.0;
        }
        SECTION("Middle") {
            split_location = 0.5;
        }
        SECTION("2/3") {
            split_location = 2.0/3.0;
        }
        SECTION("End") {
            split_location = 1.0;
        }

        const auto parts = split(curve, split_location);
        if (split_location == 0.0) {
            REQUIRE(parts.size() == 1);
            assert_same(curve, parts[0], 10);
        } else if (split_location == 1.0) {
            REQUIRE(parts.size() == 1);
            assert_same(curve, parts[0], 10);
        } else {
            assert_same(curve, parts[0], 10, 0.0, split_location, 0.0, split_location);
            assert_same(curve, parts[1], 10, split_location, 1.0, split_location, 1.0);
        }
    }

    SECTION("NURBS degree 2") {
        Eigen::Matrix<Scalar, 8, 2> ctrl_pts;
        ctrl_pts << 0.0, 0.0,
                    0.0, 1.0,
                    1.0, 1.0,
                    1.0, 0.0,
                    2.0, 0.0,
                    2.0, 1.0,
                    3.0, 1.0,
                    3.0, 0.0;
        Eigen::Matrix<Scalar, 11, 1> knots;
        knots << 0.0, 0.0, 0.0,
                 1.0, 2.0, 3.0, 4.0, 5.0,
                 6.0, 6.0, 6.0;
        Eigen::Matrix<Scalar ,8, 1> weights;
        weights << 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5;

        NURBS<Scalar, 2, 2, true> curve;
        curve.set_control_points(ctrl_pts);
        curve.set_knots(knots);
        curve.set_weights(weights);
        curve.initialize();
        //save_svg("test.svg", curve);

        Scalar split_location = 0.0;
        SECTION("Beginning") {
            split_location = 0.0;
        }
        SECTION("End") {
            split_location = 6.0;
        }
        SECTION("Half") {
            split_location = 3.0;
        }
        SECTION("Between 2 knots") {
            split_location = 0.5;
        }

        const auto parts = split(curve, split_location);
        if (split_location == 0.0) {
            REQUIRE(parts.size() == 1);
            assert_same(curve, parts[0], 10);
        } else if (split_location == 6.0) {
            REQUIRE(parts.size() == 1);
            assert_same(curve, parts[0], 10);
        } else {
            assert_same(curve, parts[0], 10, 0.0, split_location, 0.0, split_location);
            assert_same(curve, parts[1], 10, split_location, 6.0, split_location, 6.0);
        }
    }
}
