#include <catch2/catch.hpp>

#include <cmath>

#include <nanospline/RationalBezier.h>

TEST_CASE("RationalBezier", "[rational][bezier]") {
    using namespace nanospline;

    SECTION("Generic degree 0") {
        Eigen::Matrix<float, 1, 2> control_pts;
        control_pts << 0.0, 0.1;
        Eigen::Matrix<float, 1, 1> weights;
        weights << 1.0;

        RationalBezier<float, 2, 0, true> curve;
        curve.set_control_points(control_pts);
        curve.set_weights(weights);
        curve.initialize();

        auto start = curve.evaluate(0);
        auto mid = curve.evaluate(0.5);
        auto end = curve.evaluate(1);

        REQUIRE((start-control_pts.row(0)).norm() == Approx(0.0));
        REQUIRE((end-control_pts.row(0)).norm() == Approx(0.0));
        REQUIRE((mid-control_pts.row(0)).norm() == Approx(0.0));
    }

    SECTION("Generic degree 1") {
        Eigen::Matrix<float, 2, 2> control_pts;
        control_pts << 0.0, 0.0,
                       1.0, 0.0;
        Eigen::Matrix<float, 2, 1> weights;
        weights << 0.1, 1.0;

        RationalBezier<float, 2, 1, true> curve;
        curve.set_control_points(control_pts);
        curve.set_weights(weights);
        curve.initialize();

        auto start = curve.evaluate(0);
        auto mid = curve.evaluate(0.5);
        auto end = curve.evaluate(1);

        REQUIRE((start-control_pts.row(0)).norm() == Approx(0.0));
        REQUIRE((end-control_pts.row(1)).norm() == Approx(0.0));
        REQUIRE((mid-control_pts.row(0)).norm() > 0.5);
    }

    SECTION("Generic degree 2") {
        RationalBezier<float, 2, 2, true> curve;

        Eigen::Matrix<float, 3, 2> control_pts;
        control_pts << 0.0, 0.0,
                       1.0, 1.0,
                       2.0, 0.0;
        curve.set_control_points(control_pts);

        Eigen::Matrix<float, 3, 1> weights;

        SECTION("Uniform weight") {
            SECTION("all ones") {
                weights << 1.0, 1.0, 1.0;
            }
            SECTION("all twos") {
                weights << 2.0, 2.0, 2.0;
            }
            curve.set_weights(weights);
            curve.initialize();

            Bezier<float, 2, 2> regular_bezier;
            regular_bezier.set_control_points(control_pts);

            for (float t=0.0; t<1.01; t+=0.2) {
                const auto p0 = curve.evaluate(t);
                const auto p1 = regular_bezier.evaluate(t);
                REQUIRE((p0-p1).norm() == Approx(0.0));
            }
        }

        SECTION("Non-uniform weights") {
            SECTION("Positive weight") {
                weights << 1.0, 2.0, 1.0;
                curve.set_weights(weights);
                curve.initialize();

                const auto mid = curve.evaluate(0.5);
                REQUIRE(mid[1] > 0.5);
            }
            SECTION("Zero weight") {
                weights << 1.0, 0.0, 1.0;
                curve.set_weights(weights);
                curve.initialize();

                const auto mid = curve.evaluate(0.5);
                REQUIRE(mid[1] == Approx(0.0));
            }
            SECTION("Negative weight") {
                weights << 1.0, -1.0, 1.0;
                curve.set_weights(weights);
                curve.initialize();

                const auto mid = curve.evaluate(0.5);
                REQUIRE(mid[1] < 0.0);
            }

            auto start = curve.evaluate(0);
            auto end = curve.evaluate(1);

            REQUIRE((start-control_pts.row(0)).norm() == Approx(0.0));
            REQUIRE((end-control_pts.row(2)).norm() == Approx(0.0));
        }
    }

    SECTION("Circular arc") {
        // Rational quadratic Bezier is capable of representing circular arc.
        // Let's check.

        SECTION("Quarter circle") {
            const float R = 1.2;
            RationalBezier<float, 2, 2, true> curve;

            Eigen::Matrix<float, 3, 2> control_pts;
            control_pts << R, 0.0,
                           R, R,
                           0.0, R;
            curve.set_control_points(control_pts);

            Eigen::Matrix<float, 3, 1> weights;
            weights << 1.0, sqrt(2) / 2, 1.0;
            curve.set_weights(weights);
            curve.initialize();

            for (float t=0.0; t<1.01; t+=0.2) {
                const auto p = curve.evaluate(t);
                REQUIRE(p.norm() == Approx(R));
            }
        }

        SECTION("One third circle") {
            const float R = 2.5;
            Eigen::Matrix<float, 1, 2> c(0.0, R);
            RationalBezier<float, 2, 2, true> curve;

            Eigen::Matrix<float, 3, 2> control_pts;
            control_pts << 0.5 * sqrt(3) * R, 1.5*R,
                           0.0, 3*R,
                          -0.5 * sqrt(3) * R, 1.5*R;
                            
            curve.set_control_points(control_pts);

            Eigen::Matrix<float, 3, 1> weights;
            weights << 1.0, 0.5, 1.0;
            curve.set_weights(weights);
            curve.initialize();

            for (float t=0.0; t<1.01; t+=0.2) {
                const auto p = curve.evaluate(t);
                REQUIRE((p-c).norm() == Approx(R));
            }
        }
    }
}
