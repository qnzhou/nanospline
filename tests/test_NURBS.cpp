#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <iostream>
#include <nanospline/NURBS.h>
#include <nanospline/save_msh.h>
#include <nanospline/forward_declaration.h>

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
        REQUIRE_THAT(p0[0], Catch::Matchers::WithinAbs(0.0, 1e-6));
        REQUIRE_THAT(p0[1], Catch::Matchers::WithinAbs(0.0, 1e-6));

        auto p1 = curve.evaluate(0.6);
        REQUIRE_THAT(p1[0], Catch::Matchers::WithinAbs(1.0, 1e-6));
        REQUIRE_THAT(p1[1], Catch::Matchers::WithinAbs(0.0, 1e-6));

        auto p2 = curve.evaluate(1.0);
        REQUIRE_THAT(p2[0], Catch::Matchers::WithinAbs(2.0, 1e-6));
        REQUIRE_THAT(p2[1], Catch::Matchers::WithinAbs(0.0, 1e-6));

        SECTION("Derivative") {
            validate_derivatives(curve, 10);
            validate_2nd_derivatives(curve, 10);
        }

        SECTION("Approximate inverse evaluate") {
            validate_approximate_inverse_evaluation(curve, 10);
        }

        SECTION("Update") {
            offset_and_validate(curve);
        }
    }

    SECTION("degree 1") {
        Eigen::Matrix<Scalar, 3, 2> control_pts;
        control_pts << 0.0, 0.0,
                       1.0, 0.0,
                       1.0, 1.0;
        Eigen::Matrix<Scalar, 5, 1> knots;
        knots << 0.0, 0.0, 0.5, 1.0, 1.0;

        Eigen::Matrix<Scalar, 3, 1> weights;
        weights << 1.0, 1.0, 1.0;

        NURBS<Scalar, 2, 1> curve;
        curve.set_control_points(control_pts);
        curve.set_knots(knots);
        curve.set_weights(weights);
        curve.initialize();

        SECTION("Evaluation") {
            auto p0 = curve.evaluate(0.0);
            REQUIRE_THAT(p0[0], Catch::Matchers::WithinAbs(0.0, 1e-6));
            REQUIRE_THAT(p0[1], Catch::Matchers::WithinAbs(0.0, 1e-6));

            auto p1 = curve.evaluate(1.0);
            REQUIRE_THAT(p1[0], Catch::Matchers::WithinAbs(1.0, 1e-6));
            REQUIRE_THAT(p1[1], Catch::Matchers::WithinAbs(1.0, 1e-6));
        }

        SECTION("Derivative") {
            validate_derivatives(curve, 10);
            validate_2nd_derivatives(curve, 10);
        }

        SECTION("Approximate inverse evaluate") {
            validate_approximate_inverse_evaluation(curve, 10);
        }

        SECTION("Update") {
            offset_and_validate(curve);
        }
    }

    SECTION("degree 2") {
        Eigen::Matrix<Scalar, 4, 2> control_pts;
        control_pts << 0.0, 0.0,
                       1.0, 0.0,
                       1.0, 1.0,
                       0.0, 1.0;
        Eigen::Matrix<Scalar, 7, 1> knots;
        knots << 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0;

        Eigen::Matrix<Scalar, 4, 1> weights;
        weights << 1.0, 1.0, 1.0, 1.0;

        NURBS<Scalar, 2, 2> curve;
        curve.set_control_points(control_pts);
        curve.set_knots(knots);
        curve.set_weights(weights);
        curve.initialize();

        SECTION("Evaluation") {
            auto p0 = curve.evaluate(0.0);
            REQUIRE_THAT(p0[0], Catch::Matchers::WithinAbs(0.0, 1e-6));
            REQUIRE_THAT(p0[1], Catch::Matchers::WithinAbs(0.0, 1e-6));

            auto p1 = curve.evaluate(1.0);
            REQUIRE_THAT(p1[0], Catch::Matchers::WithinAbs(0.0, 1e-6));
            REQUIRE_THAT(p1[1], Catch::Matchers::WithinAbs(1.0, 1e-6));
        }

        SECTION("weights") {
            auto p = curve.evaluate(0.5);
            REQUIRE_THAT(p[1], Catch::Matchers::WithinAbs(0.5, 1e-6));

            weights[1] = 0.1;
            weights[2] = 0.1;
            curve.set_weights(weights);
            curve.initialize();

            auto q = curve.evaluate(0.5);
            REQUIRE_THAT(q[1], Catch::Matchers::WithinAbs(0.5, 1e-6));
            REQUIRE_THAT(q[0], Catch::Matchers::WithinAbs(p[0], 1e-6));
        }

        SECTION("Derivative") {
            validate_derivatives(curve, 10);
            validate_2nd_derivatives(curve, 10);
        }

        SECTION("Approximate inverse evaluate") {
            validate_approximate_inverse_evaluation(curve, 10);
        }

        SECTION("Update") {
            offset_and_validate(curve);
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

            SECTION("Inflections") {
#if NANOSPLINE_SYMPY
                auto inflections = curve.compute_inflections(0, 1);
                REQUIRE(inflections.size() == 0);
#endif
            }

            SECTION("Degree elevation") {
                const auto curve2 = curve.elevate_degree();
                assert_same(curve, curve2, 10);
            }

            SECTION("Update") {
                offset_and_validate(curve);
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
                REQUIRE_THAT((p-c).norm(), Catch::Matchers::WithinAbs(R, 1e-6));
            }

            SECTION("Derivative") {
                validate_derivatives(curve, 10);
                validate_2nd_derivatives(curve, 10);
            }
            SECTION("Approximate inverse evaluate") {
                validate_approximate_inverse_evaluation(curve, 10);
            }

            SECTION("Insert and remove knot") {
                auto curve2 = curve;
                curve2.insert_knot(0.6);
                assert_same(curve, curve2, 10);

                curve2.insert_knot(0.7, 2);
                assert_same(curve, curve2, 10);

                curve2.initialize();
                assert_same(curve, curve2, 10);

                REQUIRE(curve2.remove_knot(0.6, 1) == 1);
                assert_same(curve, curve2, 10);

                REQUIRE(curve2.remove_knot(0.7, 2) == 2);
                assert_same(curve, curve2, 10);
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

            SECTION("Inflections") {
#if NANOSPLINE_SYMPY
                auto inflections = curve.compute_inflections(0, 1);
                REQUIRE(inflections.size() == 0);
#endif
            }

            SECTION("Singularities") {
#if NANOSPLINE_SYMPY
                auto singular_pts = curve.compute_singularities(0, 1);
                REQUIRE(singular_pts.size() == 0);
#endif
            }

            SECTION("Degree elevation") {
                const auto curve2 = curve.elevate_degree();
                assert_same(curve, curve2, 10);
            }

            SECTION("Update") {
                offset_and_validate(curve);
            }
        }
    }

    SECTION("Inflection points") {
#if NANOSPLINE_SYMPY
        Eigen::Matrix<Scalar, 4, 2> ctrl_pts;
        ctrl_pts << 0.0, 0.0,
                    1.0, 1.0,
                    2.0,-1.0,
                    3.0, 0.0;
        Eigen::Matrix<Scalar, 8, 1> knots;
        knots << 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0;
        Eigen::Matrix<Scalar, 4, 1> weights;
        weights << 2.0, 2.0, 2.0, 2.0;

        NURBS<Scalar, 2, 3> curve;
        curve.set_control_points(ctrl_pts);
        curve.set_knots(knots);
        curve.set_weights(weights);
        curve.initialize();

        SECTION("Inflection") {
            auto inflections = curve.compute_inflections(0, 1);
            REQUIRE(inflections.size() == 1);
            REQUIRE_THAT(inflections[0], Catch::Matchers::WithinAbs(0.5, 1e-6));
        }

        SECTION("Singularity") {
            auto singular_pts = curve.compute_singularities(0, 1);
            REQUIRE(singular_pts.size() == 0);
        }

        SECTION("Degree elevation") {
            const auto curve2 = curve.elevate_degree();
            assert_same(curve, curve2, 10);
        }

        SECTION("Approximate inverse evaluate") {
            validate_approximate_inverse_evaluation(curve, 10);
        }

        SECTION("Update") {
            offset_and_validate(curve);
        }
#endif
    }

    SECTION("Inflection points of closed NURBS curve") {
#if NANOSPLINE_SYMPY
        Eigen::Matrix<Scalar, 14, 2> ctrl_pts;
        ctrl_pts << 1, 4, .5, 6, 5, 4, 3, 12, 11, 14, 8, 4, 12, 3, 11, 9, 15, 10, 17, 8,
                 1, 4, .5, 6, 5, 4, 3, 12 ;
        Eigen::Matrix<Scalar, 18, 1> knots;
        knots << 0.0/17, 1.0/17, 2.0/17, 3.0/17, 4.0/17, 5.0/17, 6.0/17, 7.0/17,
              8.0/17, 9.0/17, 10.0/17, 11.0/17, 12.0/17, 13.0/17, 14.0/17, 15.0/17,
              16.0/17, 17.0/17;
        Eigen::Matrix<Scalar, 14, 1> weights;
        weights.setConstant(1.2);

        NURBS<Scalar, 2, 3, true> curve;
        curve.set_control_points(ctrl_pts);
        curve.set_knots(knots);
        curve.set_weights(weights);
        curve.initialize();

        SECTION("Inflection") {
            auto inflections = curve.compute_inflections(
                    knots.minCoeff(), knots.maxCoeff());
            for (auto t : inflections) {
                auto k = curve.evaluate_curvature(t).norm();
                REQUIRE_THAT(k, Catch::Matchers::WithinAbs(0, 1e-6));
            }
        }

        SECTION("Singularity") {
            auto singular_pts = curve.compute_singularities(
                    knots.minCoeff(), knots.maxCoeff());
            REQUIRE(singular_pts.size() == 0);
        }

        SECTION("Degree elevation") {
            const auto curve2 = curve.elevate_degree();
            assert_same(curve, curve2, 10);
        }

        SECTION("Approximate inverse evaluate") {
            validate_approximate_inverse_evaluation(curve, 10);
        }

        SECTION("Update") {
            offset_and_validate(curve);
        }
#endif
    }

    SECTION("Extrapolation") {
        Eigen::Matrix<Scalar, 5, 2> control_pts;
        control_pts << 0.0, 0.0,
                       1.0, 1.0,
                       2.0, 0.5,
                       1.0, 0.0,
                       0.0, 1.0;
        NURBS<Scalar, 2, -1> curve;
        curve.set_control_points(control_pts);

        Eigen::Matrix<Scalar, 9, 1> knots;
        knots << 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 2.0;
        curve.set_knots(knots);

        Eigen::Matrix<Scalar, 5, 1> weights;
        weights << 1.0, 1.0, 0.1, 1.0, 1.0;
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

    SECTION("Turning angle debug") {
#if NANOSPLINE_SYMPY
        Eigen::Matrix<Scalar, 4, 2> control_pts;
        control_pts << 
            97.30000000000004, 6.200000000000202,
            97.30000000000032, -4.80000000000031,
            86.30000000000015, -4.800000000000409,
            86.29999999999987, 6.2000000000001165;
        Eigen::Matrix<Scalar, 4, 1> weights;
        weights << 1.0, 0.333333333333334, 0.333333333333334, 1.0;
        Eigen::Matrix<Scalar, 8, 1> knots;
        knots << 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0;

        NURBS<Scalar, 2, -1> curve;
        curve.set_control_points(control_pts);
        curve.set_weights(weights);
        curve.set_knots(knots);
        curve.initialize();

        auto theta = curve.get_turning_angle(0, 1);
        REQUIRE_THAT(std::abs(theta), Catch::Matchers::WithinAbs(M_PI, 1e-6));
        auto cuts = curve.reduce_turning_angle(0, 1);
        REQUIRE(cuts.size() == 1);

        auto theta_1 = curve.get_turning_angle(0, cuts[0]);
        auto theta_2 = curve.get_turning_angle(cuts[0], 1);
        REQUIRE_THAT(theta_1 + theta_2, Catch::Matchers::WithinAbs(theta, 1e-6));
#endif
    }

    SECTION("Periodic curve") {
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

        REQUIRE(curve.is_closed());
        REQUIRE(curve.is_closed(1, 1e-6));

        curve.set_periodic(true);
        curve.initialize();

        Eigen::Matrix<Scalar, 1, 2> q(0.0, -1.0);
        Scalar t0 = curve.approximate_inverse_evaluate(q, 0, 1);
        REQUIRE_THAT(curve.evaluate(t0).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
        Scalar t1 = curve.approximate_inverse_evaluate(q, 1.5, 2.5);
        REQUIRE_THAT(curve.evaluate(t1).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
        Scalar t2 = curve.approximate_inverse_evaluate(q, -10.1, -9.9);
        REQUIRE_THAT(curve.evaluate(t2).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
        Scalar t3 = curve.approximate_inverse_evaluate(q, 2.25, 2.75);
        REQUIRE(curve.evaluate(t3).norm() > 0.1);

        auto curve2 = curve.elevate_degree();
        REQUIRE(curve2.get_periodic());
        assert_same(curve, curve2, 10);
    }
}
