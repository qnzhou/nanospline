#include <catch2/catch.hpp>

#include <cmath>

#include <nanospline/RationalBezier.h>
#include <nanospline/save_svg.h>
#include "forward_declaration.h"
#include "validation_utils.h"

TEST_CASE("RationalBezier", "[rational][bezier]") {
    using namespace nanospline;
    using Scalar = double;

    SECTION("Generic degree 0") {
        Eigen::Matrix<Scalar, 1, 2> control_pts;
        control_pts << 0.0, 0.1;
        Eigen::Matrix<Scalar, 1, 1> weights;
        weights << 1.0;

        RationalBezier<Scalar, 2, 0, true> curve;
        curve.set_control_points(control_pts);
        curve.set_weights(weights);
        curve.initialize();

        auto start = curve.evaluate(0);
        auto mid = curve.evaluate(0.5);
        auto end = curve.evaluate(1);

        REQUIRE((start-control_pts.row(0)).norm() == Approx(0.0));
        REQUIRE((end-control_pts.row(0)).norm() == Approx(0.0));
        REQUIRE((mid-control_pts.row(0)).norm() == Approx(0.0));

        SECTION("Derivative") {
            validate_derivatives(curve, 10);
            validate_2nd_derivatives(curve, 10);
        }
        SECTION("Curvature") {
            auto k = curve.evaluate_curvature(0.4);
            REQUIRE(k.norm() == Approx(0.0));
        }
    }

    SECTION("Generic degree 1") {
        Eigen::Matrix<Scalar, 2, 2> control_pts;
        control_pts << 0.0, 0.0,
                       1.0, 0.0;
        Eigen::Matrix<Scalar, 2, 1> weights;
        weights << 0.1, 1.0;

        RationalBezier<Scalar, 2, 1, true> curve;
        curve.set_control_points(control_pts);
        curve.set_weights(weights);
        curve.initialize();

        auto start = curve.evaluate(0);
        auto mid = curve.evaluate(0.5);
        auto end = curve.evaluate(1);

        REQUIRE((start-control_pts.row(0)).norm() == Approx(0.0));
        REQUIRE((end-control_pts.row(1)).norm() == Approx(0.0));
        REQUIRE((mid-control_pts.row(0)).norm() > 0.5);

        SECTION("Derivative") {
            validate_derivatives(curve, 10);
            validate_2nd_derivatives(curve, 10);
        }
        SECTION("Curvature") {
            auto k = curve.evaluate_curvature(0.4);
            REQUIRE(k.norm() == Approx(0.0));
        }
        SECTION("Turning angle") {
            const auto total_turning_angle = curve.get_turning_angle(0, 1);
            REQUIRE(total_turning_angle == Approx(0));
            const auto split_pts = curve.reduce_turning_angle(0, 1);
            REQUIRE(split_pts.size() == 0);
        }
    }

    SECTION("Generic degree 2") {
        RationalBezier<Scalar, 2, 2, true> curve;

        Eigen::Matrix<Scalar, 3, 2> control_pts;
        control_pts << 0.0, 0.0,
                       1.0, 1.0,
                       2.0, 0.0;
        curve.set_control_points(control_pts);

        Eigen::Matrix<Scalar, 3, 1> weights;

        SECTION("Uniform weight") {
            SECTION("all ones") {
                weights << 1.0, 1.0, 1.0;
            }
            SECTION("all twos") {
                weights << 2.0, 2.0, 2.0;
            }
            curve.set_weights(weights);
            curve.initialize();

            Bezier<Scalar, 2, 2> regular_bezier;
            regular_bezier.set_control_points(control_pts);

            assert_same(curve, regular_bezier, 10);
            validate_derivatives(curve, 10);
            validate_2nd_derivatives(curve, 10);

            const auto total_turning_angle = curve.get_turning_angle(0, 1);
            REQUIRE(total_turning_angle == Approx(M_PI/2));
            const auto split_pts = curve.reduce_turning_angle(0, 1);
            REQUIRE(split_pts.size() == 1);
            const auto turning_angle_1 = curve.get_turning_angle(0, split_pts[0]);
            const auto turning_angle_2 = curve.get_turning_angle(split_pts[0], 1);
            REQUIRE(turning_angle_1 == Approx(M_PI/4));
            REQUIRE(turning_angle_2 == Approx(M_PI/4));
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
            validate_derivatives(curve, 10, 1e-5);
            validate_2nd_derivatives(curve, 10);

        }
    }

    SECTION("Circular arc") {
        // Rational quadratic Bezier is capable of representing circular arc.
        // Let's check.

        SECTION("Quarter circle") {
            const Scalar R = 1.2;
            RationalBezier<Scalar, 2, 2, true> curve;

            Eigen::Matrix<Scalar, 3, 2> control_pts;
            control_pts << R, 0.0,
                           R, R,
                           0.0, R;
            curve.set_control_points(control_pts);

            Eigen::Matrix<Scalar, 3, 1> weights;
            weights << 1.0, sqrt(2) / 2, 1.0;
            curve.set_weights(weights);
            curve.initialize();

            for (Scalar t=0.0; t<1.01; t+=0.2) {
                const auto p = curve.evaluate(t);
                REQUIRE(p.norm() == Approx(R));
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
            SECTION("Turning angle") {
                const auto total_turning_angle = curve.get_turning_angle(0, 1);
                REQUIRE(total_turning_angle == Approx(M_PI/2));
                const auto split_pts = curve.reduce_turning_angle(0, 1);
                REQUIRE(split_pts.size() == 1);
                const auto turning_angle_1 = curve.get_turning_angle(0, split_pts[0]);
                const auto turning_angle_2 = curve.get_turning_angle(split_pts[0], 1);
                REQUIRE(turning_angle_1 == Approx(M_PI/4));
                REQUIRE(turning_angle_2 == Approx(M_PI/4));
            }
        }

        SECTION("One third circle") {
            const Scalar R = 2.5;
            Eigen::Matrix<Scalar, 1, 2> c(0.0, R);
            RationalBezier<Scalar, 2, 2, true> curve;

            Eigen::Matrix<Scalar, 3, 2> control_pts;
            control_pts << 0.5 * sqrt(3) * R, 1.5*R,
                           0.0, 3*R,
                          -0.5 * sqrt(3) * R, 1.5*R;

            curve.set_control_points(control_pts);

            Eigen::Matrix<Scalar, 3, 1> weights;
            weights << 1.0, 0.5, 1.0;
            curve.set_weights(weights);
            curve.initialize();

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
            SECTION("Turning angle") {
                const auto total_turning_angle = curve.get_turning_angle(0, 1);
                REQUIRE(total_turning_angle == Approx(2*M_PI/3));
                const auto split_pts = curve.reduce_turning_angle(0, 1);
                REQUIRE(split_pts.size() == 1);
                const auto turning_angle_1 = curve.get_turning_angle(0, split_pts[0]);
                const auto turning_angle_2 = curve.get_turning_angle(split_pts[0], 1);
                REQUIRE(turning_angle_1 == Approx(2*M_PI/6));
                REQUIRE(turning_angle_2 == Approx(2*M_PI/6));
            }
        }

        SECTION("Debug for Yixin") {
            Eigen::Matrix<Scalar, 3, 2> ctrl_pts[4];
            ctrl_pts[0] << 251.49155000000002, 185.0696,
                        272.02322999999996, 188.40176,
                        275.35539, 167.87008;

            ctrl_pts[1] << 275.35539, 167.87008,
                        278.68754999999993, 147.3384,
                        258.15587, 144.00624000000002;

            ctrl_pts[2] << 258.15587, 144.00624000000002,
                        237.62418973609834, 140.6740795131014,
                        234.29202964246906, 161.20575991329454;

            ctrl_pts[3] << 234.29202964246906, 161.20575991329454,
                        230.95986954883978, 181.73744031348764,
                        251.49155000000002, 185.0696;
            Eigen::Matrix<Scalar, 3, 1> weights;
            weights << 1.0, 0.7071067742171676, 1.0;

            std::vector<RationalBezier<Scalar, 2, 2>> arcs(4);
            arcs[0].set_control_points(ctrl_pts[0]);
            arcs[1].set_control_points(ctrl_pts[1]);
            arcs[2].set_control_points(ctrl_pts[2]);
            arcs[3].set_control_points(ctrl_pts[3]);

            arcs[0].set_weights(weights);
            arcs[1].set_weights(weights);
            arcs[2].set_weights(weights);
            arcs[3].set_weights(weights);

            arcs[0].initialize();
            arcs[1].initialize();
            arcs[2].initialize();
            arcs[3].initialize();

            Scalar last = 0;
            Eigen::Matrix<Scalar, 2, 1> center;
            for (const auto& curve : arcs) {
                auto k = curve.evaluate_curvature(0.5);
                if (last == 0) {
                    last = k.norm();
                    center = curve.evaluate(0.5) + k / k.squaredNorm();
                } else {
                    REQUIRE(last == Approx(k.norm()));
                    Eigen::Matrix<Scalar, 2, 1> c = curve.evaluate(0.5) + k / k.squaredNorm();
                    REQUIRE((c-center).norm() == Approx(0).margin(1e-6));
                }
            }
        }
    }
}
