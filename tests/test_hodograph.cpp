#include <catch2/catch.hpp>

#include <nanospline/hodograph.h>
#include "validation_utils.h"

TEST_CASE("hodograph", "[hodograph]") {
    using namespace nanospline;
    using Scalar = double;

    SECTION("Bezier degree 3") {
        Eigen::Matrix<Scalar, 4, 2> ctrl_pts;
        ctrl_pts << 0.0, 0.0,
                    0.0, 1.0,
                    1.0, 1.0,
                    1.0, 0.0;

        Bezier<Scalar, 2, 3> curve;
        curve.set_control_points(ctrl_pts);

        auto hodograph = compute_hodograph(curve);
        REQUIRE(hodograph.get_degree() == 2);
        validate_hodograph(curve, hodograph, 10);

        auto hodograph2 = compute_hodograph(hodograph);
        REQUIRE(hodograph2.get_degree() == 1);
        validate_hodograph(hodograph, hodograph2, 10);

        auto hodograph3 = compute_hodograph(hodograph2);
        REQUIRE(hodograph3.get_degree() == 0);
        validate_hodograph(hodograph2, hodograph3, 10);

        auto hodograph4 = compute_hodograph(hodograph3);
        REQUIRE(hodograph4.get_degree() == 0);
        assert_same(hodograph3, hodograph4, 10);
    }

    SECTION("BSpline degree 1") {
        Eigen::Matrix<Scalar, 4, 2> ctrl_pts;
        ctrl_pts << 0.0, 0.0,
                    0.0, 1.0,
                    1.0, 1.0,
                    1.0, 0.0;
        Eigen::Matrix<Scalar, 6, 1> knots;
        knots << 0, 0, 1, 2, 3, 3;

        BSpline<Scalar, 2, 1, true> curve;
        curve.set_control_points(ctrl_pts);
        curve.set_knots(knots);

        auto hodograph = compute_hodograph(curve);
        REQUIRE(hodograph.get_degree() == 0);
        validate_hodograph(curve, hodograph, 10);

        SECTION("Recursive hodograph") {
            auto hodograph2 = compute_hodograph(hodograph);
            assert_same(hodograph, hodograph2, 10);
        }
    }

    SECTION("BSpline degree 1 dynamic") {
        Eigen::Matrix<Scalar, 4, 2> ctrl_pts;
        ctrl_pts << 0.0, 0.0,
                    0.0, 1.0,
                    1.0, 1.0,
                    1.0, 0.0;
        Eigen::Matrix<Scalar, 6, 1> knots;
        knots << 0, 0, 1, 2, 3, 3;

        BSpline<Scalar, 2, -1> curve;
        curve.set_control_points(ctrl_pts);
        curve.set_knots(knots);

        auto hodograph = compute_hodograph(curve);
        REQUIRE(hodograph.get_degree() == 0);
        validate_hodograph(curve, hodograph, 10);

        SECTION("Recursive hodograph") {
            auto hodograph2 = compute_hodograph(hodograph);
            assert_same(hodograph, hodograph2, 10);
        }
    }

    SECTION("BSpline degree 2") {
        Eigen::Matrix<Scalar, 4, 2> ctrl_pts;
        ctrl_pts << 0.0, 0.0,
                    0.0, 1.0,
                    1.0, 1.0,
                    1.0, 0.0;
        Eigen::Matrix<Scalar, 7, 1> knots;
        knots << 0, 0, 0, 1, 2, 2, 2;

        BSpline<Scalar, 2, 2, true> curve;
        curve.set_control_points(ctrl_pts);
        curve.set_knots(knots);

        auto hodograph = compute_hodograph(curve);
        REQUIRE(hodograph.get_degree() == 1);
        validate_hodograph(curve, hodograph, 10);
    }

    SECTION("BSpline degree 3") {
        Eigen::Matrix<Scalar, 10, 2> ctrl_pts;
        ctrl_pts << 1, 4, .5, 6, 5, 4, 3, 12, 11, 14, 8, 4, 12, 3, 11, 9, 15, 10, 17, 8;
        Eigen::Matrix<Scalar, 14, 1> knots;
        knots << 0, 0, 0, 0, 1.0/7, 2.0/7, 3.0/7, 4.0/7, 5.0/7, 6.0/7, 1, 1, 1, 1;

        BSpline<Scalar, 2, 3, true> curve;
        curve.set_control_points(ctrl_pts);
        curve.set_knots(knots);

        auto hodograph = compute_hodograph(curve);
        REQUIRE(hodograph.get_degree() == 2);
        validate_hodograph(curve, hodograph, 10);
    }

    SECTION("BSpline degree 3 dynamic") {
        Eigen::Matrix<Scalar, 10, 2> ctrl_pts;
        ctrl_pts << 1, 4, .5, 6, 5, 4, 3, 12, 11, 14, 8, 4, 12, 3, 11, 9, 15, 10, 17, 8;
        Eigen::Matrix<Scalar, 14, 1> knots;
        knots << 0, 0, 0, 0, 1.0/7, 2.0/7, 3.0/7, 4.0/7, 5.0/7, 6.0/7, 1, 1, 1, 1;

        BSpline<Scalar, 2, -1> curve;
        curve.set_control_points(ctrl_pts);
        curve.set_knots(knots);

        auto hodograph = compute_hodograph(curve);
        REQUIRE(hodograph.get_degree() == 2);
        validate_hodograph(curve, hodograph, 10);
    }
}
