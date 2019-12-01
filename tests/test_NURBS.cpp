#include <catch2/catch.hpp>

#include <cmath>
#include <iostream>
#include <nanospline/NURBS.h>
#include <nanospline/save_svg.h>

#include "validation_utils.h"

TEST_CASE("NURBS", "[rational][nurbs][bspline]") {
    using namespace nanospline;
    using Scalar = double;

    SECTION("Generic degree 0") {
        Eigen::Matrix<Scalar, 3, 2> control_pts;
        control_pts << 0.0, 0.0,
                       1.0, 0.0,
                       2.0, 0.0;
        Eigen::Matrix<Scalar, 4, 1> knots;
        knots << 0.0, 0.5, 0.75, 1.0;

        Eigen::Matrix<Scalar, 3, 1> weights;
        weights << 1.0, 2.0, 3.0;

        NURBS<Scalar, 2, 0, true> curve;
        curve.set_control_points(control_pts);
        curve.set_knots(knots);
        curve.set_weights(weights);
        curve.initialize();

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

        SECTION("Derivative") {
            validate_derivatives(curve, 10);
            validate_2nd_derivatives(curve, 10);
        }
    }

    SECTION("Circles") {
        SECTION("3-way split") {
            constexpr Scalar R = 1.1;
            Eigen::Matrix<Scalar, 1, 2> c(0.0, R);
            Eigen::Matrix<Scalar, 7, 2> control_pts;
            control_pts << 0.0, 0.0,
                           sqrt(3)*R, 0.0,
                           0.5*sqrt(3)*R, 1.5*R,
                           0.0, 3*R,
                          -0.5*sqrt(3)*R, 1.5*R,
                          -sqrt(3)*R, 0.0,
                           0.0, 0.0;
            Eigen::Matrix<Scalar, 10, 1> knots;
            knots << 0.0, 0.0, 0.0,
                     1.0/3.0, 1.0/3.0,
                     2.0/3.0, 2.0/3.0,
                     1.0, 1.0, 1.0;

            Eigen::Matrix<Scalar, 7, 1> weights;
            weights << 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0;

            NURBS<Scalar, 2, 2, true> curve;
            curve.set_control_points(control_pts);
            curve.set_knots(knots);
            curve.set_weights(weights);
            curve.initialize();

            REQUIRE(curve.get_degree() == 2);

            for (Scalar t=0.0; t<1.01; t+=0.2) {
                const auto p = curve.evaluate(t);
                REQUIRE((p-c).norm() == Approx(R));
            }
            SECTION("Derivative") {
                validate_derivatives(curve, 10);
                validate_2nd_derivatives(curve, 10);
            }
            SECTION("Curvature") {
                auto k = curve.evaluate_curvature(0.4);
                REQUIRE(k.norm() == Approx(1.0/R));
                k = curve.evaluate_curvature(0.5);
                REQUIRE(k.norm() == Approx(1.0/R));
                k = curve.evaluate_curvature(0.8);
                REQUIRE(k.norm() == Approx(1.0/R));
                k = curve.evaluate_curvature(0.0);
                REQUIRE(k.norm() == Approx(1.0/R));
                k = curve.evaluate_curvature(1.0);
                REQUIRE(k.norm() == Approx(1.0/R));
            }
        }

        SECTION("4-way split") {
            constexpr Scalar R = 12.1;
            Eigen::Matrix<Scalar, 1, 2> c(0.0, R);
            Eigen::Matrix<Scalar, 9, 2> control_pts;
            control_pts << 0.0, 0.0,
                           R, 0.0,
                           R, R,
                           R, 2*R,
                           0.0, 2*R,
                           -R, 2*R,
                           -R, R,
                           -R, 0.0,
                           0.0, 0.0;
            Eigen::Matrix<Scalar, 12, 1> knots;
            knots << 0.0, 0.0, 0.0,
                     0.25, 0.25,
                     0.5, 0.5,
                     0.75, 0.75,
                     1.0, 1.0, 1.0;
            Eigen::Matrix<Scalar, 9, 1> weights;
            weights << 1.0, sqrt(2)/2,
                       1.0, sqrt(2)/2,
                       1.0, sqrt(2)/2,
                       1.0, sqrt(2)/2,
                       1.0;

            NURBS<Scalar, 2, 2, true> curve;
            curve.set_control_points(control_pts);
            curve.set_knots(knots);
            curve.set_weights(weights);
            curve.initialize();

            REQUIRE(curve.get_degree() == 2);

            for (Scalar t=0.0; t<1.01; t+=0.2) {
                const auto p = curve.evaluate(t);
                REQUIRE((p-c).norm() == Approx(R));
            }

            SECTION("Derivative") {
                validate_derivatives(curve, 10);
                validate_2nd_derivatives(curve, 10);
            }

            SECTION("Insert knot") {
                auto curve2 = curve;
                curve2.insert_knot(0.6);
                assert_same(curve, curve2, 10);

                curve2.insert_knot(0.7, 2);
                assert_same(curve, curve2, 10);

                curve2.initialize();
                assert_same(curve, curve2, 10);
            }

            SECTION("Curvature") {
                auto k = curve.evaluate_curvature(0.4);
                REQUIRE(k.norm() == Approx(1.0/R));
                k = curve.evaluate_curvature(0.5);
                REQUIRE(k.norm() == Approx(1.0/R));
                k = curve.evaluate_curvature(0.8);
                REQUIRE(k.norm() == Approx(1.0/R));
                k = curve.evaluate_curvature(0.0);
                REQUIRE(k.norm() == Approx(1.0/R));
                k = curve.evaluate_curvature(1.0);
                REQUIRE(k.norm() == Approx(1.0/R));
            }
        }
    }
}
