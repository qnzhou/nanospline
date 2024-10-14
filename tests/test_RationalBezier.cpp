#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include <nanospline/RationalBezier.h>
#include <nanospline/save_svg.h>
#include <nanospline/forward_declaration.h>

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

        REQUIRE_THAT((start-control_pts.row(0)).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
        REQUIRE_THAT((end-control_pts.row(0)).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
        REQUIRE_THAT((mid-control_pts.row(0)).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));

        SECTION("Derivative") {
            validate_derivatives(curve, 10);
            validate_2nd_derivatives(curve, 10);
        }
        SECTION("Curvature") {
            auto k = curve.evaluate_curvature(0.4);
            REQUIRE_THAT(k.norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
        }
        SECTION("Degree elevation") {
            const auto new_curve = curve.elevate_degree();
            REQUIRE(new_curve.get_degree() == curve.get_degree()+1);
            assert_same(curve, new_curve, 10);
        }
        SECTION("Approximate inverse evaluate") {
            validate_approximate_inverse_evaluation(curve, 10);
        }
        SECTION("Update") {
            offset_and_validate(curve);
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

        REQUIRE_THAT((start-control_pts.row(0)).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
        REQUIRE_THAT((end-control_pts.row(1)).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
        REQUIRE((mid-control_pts.row(0)).norm() > 0.5);

        SECTION("Derivative") {
            validate_derivatives(curve, 10);
            validate_2nd_derivatives(curve, 10);
        }
        SECTION("Curvature") {
            auto k = curve.evaluate_curvature(0.4);
            REQUIRE_THAT(k.norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
        }
        SECTION("Turning angle") {
#if NANOSPLINE_SYMPY
            const auto total_turning_angle = curve.get_turning_angle(0, 1);
            REQUIRE_THAT(total_turning_angle, Catch::Matchers::WithinAbs(0.0, 1e-6));
            const auto split_pts = curve.reduce_turning_angle(0, 1);
            REQUIRE(split_pts.size() == 0);
#endif
        }
        SECTION("Degree elevation") {
            const auto new_curve = curve.elevate_degree();
            REQUIRE(new_curve.get_degree() == curve.get_degree()+1);
            assert_same(curve, new_curve, 10);
        }
        SECTION("Approximate inverse evaluate") {
            validate_approximate_inverse_evaluation(curve, 10);
        }
        SECTION("Update") {
            offset_and_validate(curve);
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
            validate_approximate_inverse_evaluation(curve, 10);

#if NANOSPLINE_SYMPY
            const auto total_turning_angle = curve.get_turning_angle(0, 1);
            REQUIRE_THAT(std::abs(total_turning_angle), Catch::Matchers::WithinAbs(M_PI/2, 1e-6));
            const auto split_pts = curve.reduce_turning_angle(0, 1);
            REQUIRE(split_pts.size() == 1);
            const auto turning_angle_1 = curve.get_turning_angle(0, split_pts[0]);
            const auto turning_angle_2 = curve.get_turning_angle(split_pts[0], 1);
            REQUIRE_THAT(std::abs(turning_angle_1), Catch::Matchers::WithinAbs(M_PI/4, 1e-6));
            REQUIRE_THAT(std::abs(turning_angle_2), Catch::Matchers::WithinAbs(M_PI/4, 1e-6));

            const auto singular_pts = curve.compute_singularities(0, 1);
            REQUIRE(singular_pts.size() == 0);
#endif

            const auto new_curve = curve.elevate_degree();
            REQUIRE(new_curve.get_degree() == curve.get_degree()+1);
            assert_same(curve, new_curve, 10);
            offset_and_validate(curve);
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
                REQUIRE_THAT((mid[1]), Catch::Matchers::WithinAbs(0.0, 1e-6));

                auto start = curve.evaluate(0);
                auto end = curve.evaluate(1);

                REQUIRE_THAT((start-control_pts.row(0)).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
                REQUIRE_THAT((end-control_pts.row(2)).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
            }
            validate_derivatives(curve, 10, 1e-5);
            validate_2nd_derivatives(curve, 10);
            validate_approximate_inverse_evaluation(curve, 10);

#if NANOSPLINE_SYMPY
            const auto split_pts = curve.reduce_turning_angle(0, 1);
            if (weights[1] > 0) {
                REQUIRE(split_pts.size() == 1);
                const auto total_turning_angle = curve.get_turning_angle(0, 1);
                const auto turning_angle_1 = curve.get_turning_angle(0, split_pts[0]);
                const auto turning_angle_2 = curve.get_turning_angle(split_pts[0], 1);
                REQUIRE_THAT(turning_angle_1, Catch::Matchers::WithinAbs(0.5 * total_turning_angle, 1e-6));
                REQUIRE_THAT(turning_angle_2, Catch::Matchers::WithinAbs(0.5 * total_turning_angle, 1e-6));
            }
            const auto singular_pts = curve.compute_singularities(0, 1+1e-3);
            if (weights[1] == 0) {
                REQUIRE(singular_pts.size() == 2);
                for (auto t : singular_pts) {
                    const auto d = curve.evaluate_derivative(t);
                    REQUIRE_THAT(d.norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
                }
            } else {
                REQUIRE(singular_pts.size() == 0);
            }
#endif

            const auto new_curve = curve.elevate_degree();
            REQUIRE(new_curve.get_degree() == curve.get_degree()+1);
            assert_same(curve, new_curve, 10);
            offset_and_validate(curve);
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
                REQUIRE_THAT(p.norm(), Catch::Matchers::WithinAbs(R, 1e-6));
            }
            SECTION("Derivative") {
                validate_derivatives(curve, 10);
                validate_2nd_derivatives(curve, 10);
            }
            SECTION("Approximate inverse evaluate") {
                validate_approximate_inverse_evaluation(curve, 10);
            }
            SECTION("Curvature") {
                auto k = curve.evaluate_curvature(0.4);
                REQUIRE_THAT(k.norm(), Catch::Matchers::WithinAbs(1.0/R, 1e-6));
                k = curve.evaluate_curvature(0.5);
                REQUIRE_THAT(k.norm(), Catch::Matchers::WithinAbs(1.0/R, 1e-6));
                k = curve.evaluate_curvature(0.8);
                REQUIRE_THAT(k.norm(), Catch::Matchers::WithinAbs(1.0/R, 1e-6));
                k = curve.evaluate_curvature(0.0);
                REQUIRE_THAT(k.norm(), Catch::Matchers::WithinAbs(1.0/R, 1e-6));
                k = curve.evaluate_curvature(1.0);
                REQUIRE_THAT(k.norm(), Catch::Matchers::WithinAbs(1.0/R, 1e-6));
            }
            SECTION("Turning angle") {
#if NANOSPLINE_SYMPY
                const auto total_turning_angle = curve.get_turning_angle(0, 1);
                REQUIRE_THAT(std::abs(total_turning_angle), Catch::Matchers::WithinAbs(M_PI/2, 1e-6));
                const auto split_pts = curve.reduce_turning_angle(0, 1);
                REQUIRE(split_pts.size() == 1);
                const auto turning_angle_1 = curve.get_turning_angle(0, split_pts[0]);
                const auto turning_angle_2 = curve.get_turning_angle(split_pts[0], 1);
                REQUIRE_THAT(std::abs(turning_angle_1), Catch::Matchers::WithinAbs(M_PI/4, 1e-6));
                REQUIRE_THAT(std::abs(turning_angle_2), Catch::Matchers::WithinAbs(M_PI/4, 1e-6));
#endif
            }
            SECTION("Degree elevation") {
                const auto new_curve = curve.elevate_degree();
                REQUIRE(new_curve.get_degree() == curve.get_degree()+1);
                assert_same(curve, new_curve, 10);
            }
            SECTION("Update") {
                offset_and_validate(curve);
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
                REQUIRE_THAT((p-c).norm(), Catch::Matchers::WithinAbs(R, 1e-6));
            }
            SECTION("Derivative") {
                validate_derivatives(curve, 10);
                validate_2nd_derivatives(curve, 10);
            }
            SECTION("Approximate inverse evaluate") {
                validate_approximate_inverse_evaluation(curve, 10);
            }
            SECTION("Curvature") {
                auto k = curve.evaluate_curvature(0.4);
                REQUIRE_THAT(k.norm(), Catch::Matchers::WithinAbs(1.0/R, 1e-6));
                k = curve.evaluate_curvature(0.5);
                REQUIRE_THAT(k.norm(), Catch::Matchers::WithinAbs(1.0/R, 1e-6));
                k = curve.evaluate_curvature(0.8);
                REQUIRE_THAT(k.norm(), Catch::Matchers::WithinAbs(1.0/R, 1e-6));
                k = curve.evaluate_curvature(0.0);
                REQUIRE_THAT(k.norm(), Catch::Matchers::WithinAbs(1.0/R, 1e-6));
                k = curve.evaluate_curvature(1.0);
                REQUIRE_THAT(k.norm(), Catch::Matchers::WithinAbs(1.0/R, 1e-6));
            }
            SECTION("Turning angle") {
#if NANOSPLINE_SYMPY
                const auto total_turning_angle = curve.get_turning_angle(0, 1);
                REQUIRE_THAT(std::abs(total_turning_angle), Catch::Matchers::WithinAbs(2*M_PI/3, 1e-6));
                const auto split_pts = curve.reduce_turning_angle(0, 1);
                REQUIRE(split_pts.size() == 1);
                const auto turning_angle_1 = curve.get_turning_angle(0, split_pts[0]);
                const auto turning_angle_2 = curve.get_turning_angle(split_pts[0], 1);
                REQUIRE_THAT(std::abs(turning_angle_1), Catch::Matchers::WithinAbs(2*M_PI/6, 1e-6));
                REQUIRE_THAT(std::abs(turning_angle_2), Catch::Matchers::WithinAbs(2*M_PI/6, 1e-6));
#endif
            }
            SECTION("Degree elevation") {
                const auto new_curve = curve.elevate_degree();
                REQUIRE(new_curve.get_degree() == curve.get_degree()+1);
                assert_same(curve, new_curve, 10);
            }
            SECTION("Update") {
                offset_and_validate(curve);
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
                    REQUIRE_THAT(last, Catch::Matchers::WithinAbs(k.norm(), 1e-6));
                    Eigen::Matrix<Scalar, 2, 1> c = curve.evaluate(0.5) + k / k.squaredNorm();
                    REQUIRE_THAT((c-center).norm(), Catch::Matchers::WithinAbs(0, 1e-6));
                }
            }
        }
    }

    SECTION("Extrapolation") {
        Eigen::Matrix<Scalar, 4, 2> control_pts;
        control_pts << 0.0, 0.0,
                       1.0, 1.0,
                       1.0, 0.0,
                       0.0, 1.0;
        RationalBezier<Scalar, 2, -1> curve;
        curve.set_control_points(control_pts);
        Eigen::Matrix<Scalar, 4, 1> weights;
        weights << 1.0, 0.5, 0.5, 1.0;
        curve.set_weights(weights);
        curve.initialize();

        auto p0 = curve.evaluate(curve.get_domain_lower_bound() - 0.1);
        auto p1 = curve.evaluate(curve.get_domain_upper_bound() + 0.1);
        REQUIRE(p0[0] < 0);
        REQUIRE(p0[1] < 0);
        REQUIRE(p1[0] < 0);
        REQUIRE(p1[1] > 1);

        auto d0 = curve.evaluate_derivative(curve.get_domain_lower_bound() - 0.1);
        auto d1 = curve.evaluate_derivative(curve.get_domain_upper_bound() + 0.1);
        REQUIRE(d0[0] > 0);
        REQUIRE(d0[1] > 0);
        REQUIRE(d1[0] < 0);
        REQUIRE(d1[1] > 0);

        auto dd0 = curve.evaluate_2nd_derivative(curve.get_domain_lower_bound() - 0.1);
        auto dd1 = curve.evaluate_2nd_derivative(curve.get_domain_upper_bound() + 0.1);
        REQUIRE_THAT(dd0[0], Catch::Matchers::WithinAbs(dd1[0], 1e-6));
        REQUIRE_THAT(dd0[1], Catch::Matchers::WithinAbs(-dd1[1], 1e-6));
    }

    SECTION("Periodic curve") {
        Eigen::Matrix<Scalar, 4, 2> control_pts;
        control_pts << 0.0, 0.0,
                       1.0, 1.0,
                       1.0, 0.0,
                       0.0, 0.0;
        RationalBezier<Scalar, 2, -1> curve;
        curve.set_control_points(control_pts);
        Eigen::Matrix<Scalar, 4, 1> weights;
        weights << 1.0, 0.5, 0.5, 1.0;
        curve.set_weights(weights);
        curve.initialize();

        REQUIRE(curve.is_closed());
        REQUIRE(!curve.is_closed(1));
        REQUIRE(!curve.is_closed(2));

        curve.set_periodic(true);
        curve.initialize();

        Eigen::Matrix<Scalar, 1, 2> q(-1, -1);
        Scalar t0 = curve.approximate_inverse_evaluate(q, 0, 1);
        REQUIRE_THAT(curve.evaluate(t0).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
        Scalar t1 = curve.approximate_inverse_evaluate(q, 1.5, 2.5, 10);
        REQUIRE_THAT(curve.evaluate(t1).norm(), Catch::Matchers::WithinAbs(0.0, 1e-3));
        Scalar t2 = curve.approximate_inverse_evaluate(q, -2.1, -1.5, 10);
        REQUIRE_THAT(curve.evaluate(t2).norm(), Catch::Matchers::WithinAbs(0.0, 1e-3));
        Scalar t3 = curve.approximate_inverse_evaluate(q, 2.25, 2.75);
        REQUIRE(curve.evaluate(t3).norm() > 0.1);

        auto curve2 = curve.elevate_degree();
        REQUIRE(curve2.get_periodic());
        assert_same(curve, curve2, 10);
    }
}
