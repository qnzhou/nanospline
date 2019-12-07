#include <catch2/catch.hpp>

#include <nanospline/conversion.h>
#include <nanospline/save_svg.h>
#include "validation_utils.h"

TEST_CASE("conversion", "[conversion]") {
    using namespace nanospline;
    using Scalar = double;

    SECTION("BSpline to Bezier") {
        Eigen::Matrix<Scalar, 10, 2> control_pts;
        control_pts << 0.0, 0.0,
                       0.0, 1.0,
                       1.0, 1.0,
                       1.0, 0.0,
                       2.0, 0.0,
                       2.0, 1.0,
                       3.0, 1.0,
                       3.0, 0.0,
                       4.0, 0.0,
                       4.0, 1.0;
        Eigen::Matrix<Scalar, 14, 1> knots;
        knots << 0.0, 0.0, 0.0, 0.0,
                 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                 7.0, 7.0, 7.0, 7.0;
        BSpline<Scalar, 2, -1> curve;
        curve.set_control_points(control_pts);
        curve.set_knots(knots);

        auto segments = convert_to_Bezier(curve);
        REQUIRE(segments.size() == 7);

        assert_same(curve, segments[0], 10, 0.0, 1.0, 0.0, 1.0);
        assert_same(curve, segments[1], 10, 1.0, 2.0, 0.0, 1.0);
        assert_same(curve, segments[2], 10, 2.0, 3.0, 0.0, 1.0);
        assert_same(curve, segments[3], 10, 3.0, 4.0, 0.0, 1.0);
        assert_same(curve, segments[4], 10, 4.0, 5.0, 0.0, 1.0);
        assert_same(curve, segments[5], 10, 5.0, 6.0, 0.0, 1.0);
        assert_same(curve, segments[6], 10, 6.0, 7.0, 0.0, 1.0);
    }

    SECTION("Closed curve") {
        Eigen::Matrix<Scalar, 14, 2> ctrl_pts;
        ctrl_pts << 1, 4, .5, 6, 5, 4, 3, 12, 11, 14, 8, 4, 12, 3, 11, 9, 15, 10, 17, 8,
                 1, 4, .5, 6, 5, 4, 3, 12 ;
        Eigen::Matrix<Scalar, 18, 1> knots;
        knots << 0.0/17, 1.0/17, 2.0/17, 3.0/17, 4.0/17, 5.0/17, 6.0/17, 7.0/17,
              8.0/17, 9.0/17, 10.0/17, 11.0/17, 12.0/17, 13.0/17, 14.0/17, 15.0/17,
              16.0/17, 17.0/17;

        BSpline<Scalar, 2, 3, true> curve;
        curve.set_control_points(ctrl_pts);
        curve.set_knots(knots);

        SECTION("To Bezier") {
            auto segments = convert_to_Bezier(curve);
            REQUIRE(segments.size() == 11);
            for (int i=0; i<11; i++) {
                assert_same(curve, segments[i], 10, (3.0+i)/17, (4.0+i)/17, 0.0, 1.0);
            }
        }

        SECTION("To NURBS") {
            auto nurbs = convert_to_NURBS(curve);
            assert_same(curve, nurbs, 10);

            auto bspline = convert_to_BSpline(nurbs);
            assert_same(curve, bspline, 10);
        }
    }

    SECTION("Rational Bezier to Bezier") {
        Eigen::Matrix<Scalar, 4, 2> ctrl_pts;
        ctrl_pts << 0.0, 0.0,
                    0.0, 1.0,
                    1.0, 1.0,
                    1.0, 0.0;
        Eigen::Matrix<Scalar, 4, 1> weights;
        weights.setConstant(2.0);

        RationalBezier<Scalar, 2, 3> curve;
        curve.set_control_points(ctrl_pts);
        curve.set_weights(weights);
        curve.initialize();

        auto bezier = convert_to_Bezier(curve);
        assert_same(curve, bezier, 10);

        auto bspline = convert_to_BSpline(bezier);
        assert_same(curve, bspline, 10);
    }
}
