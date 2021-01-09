#include <catch2/catch.hpp>
#include <iostream>

#include <nanospline/Bezier.h>
#include <nanospline/save_svg.h>
#include <nanospline/forward_declaration.h>

#include "validation_utils.h"

TEST_CASE("Bezier", "[nonrational][bezier]") {
    using namespace nanospline;
    using Scalar = double;

    SECTION("Generic degree 0") {
        Eigen::Matrix<Scalar, 1, 2> control_pts;
        control_pts << 0.0, 0.1;
        Bezier<Scalar, 2, 0, true> curve;
        curve.set_control_points(control_pts);

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

        SECTION("degree elevation") {
            auto new_curve = curve.elevate_degree();
            REQUIRE(new_curve.get_degree() == 1);
            assert_same(curve, new_curve, 10);
        }

        SECTION("Approximate inverse evaluation") {
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
        Bezier<Scalar, 2, 1, true> curve;
        curve.set_control_points(control_pts);

        auto start = curve.evaluate(0);
        auto mid = curve.evaluate(0.5);
        auto end = curve.evaluate(1);

        REQUIRE(start[0] == Approx(0.0));
        REQUIRE(mid[0] == Approx(0.5));
        REQUIRE(end[0] == Approx(1.0));

        REQUIRE(start[1] == Approx(0.0));
        REQUIRE(mid[1] == Approx(0.0));
        REQUIRE(end[1] == Approx(0.0));

        SECTION("Derivative") {
            validate_derivatives(curve, 10);
            validate_2nd_derivatives(curve, 10);
        }

        SECTION("degree elevation") {
            auto new_curve = curve.elevate_degree();
            REQUIRE(new_curve.get_degree() == 2);
            assert_same(curve, new_curve, 10);
        }

        SECTION("Approximate inverse evaluation") {
            validate_approximate_inverse_evaluation(curve, 10);
        }

        SECTION("Update") {
            offset_and_validate(curve);
        }
    }

    SECTION("Generic degree 3") {
        Eigen::Matrix<Scalar, 4, 2> control_pts;
        control_pts << 0.0, 0.0,
                       1.0, 1.0,
                       2.0, 1.0,
                       3.0, 0.0;
        Bezier<Scalar, 2, 3, true> curve;
        curve.set_control_points(control_pts);

        SECTION("Ends") {
            auto start = curve.evaluate(0);
            auto end = curve.evaluate(1);

            REQUIRE((start-control_pts.row(0)).norm() == Approx(0.0));
            REQUIRE((end-control_pts.row(3)).norm() == Approx(0.0));
        }

        SECTION("Mid point") {
            auto p = curve.evaluate(0.5);
            REQUIRE(p[0] == Approx(1.5));
            REQUIRE(p[1] > 0.0);
            REQUIRE(p[1] < 1.0);
        }

        SECTION("Inverse evaluation") {
            Eigen::Matrix<Scalar, 1, 2> p(0.0, 1.0);
            REQUIRE_THROWS(curve.inverse_evaluate(p));
        }

        SECTION("Approximate inverse evaluation") {
            validate_approximate_inverse_evaluation(curve, 10);
        }

        SECTION("Derivative") {
            validate_derivatives(curve, 10);
            validate_2nd_derivatives(curve, 10);
        }

        SECTION("Turning angle") {
#if NANOSPLINE_SYMPY
            const auto total_turning_angle = curve.get_turning_angle(0, 1);
            REQUIRE(std::abs(total_turning_angle) == Approx(M_PI/2));
            const auto split_pts = curve.reduce_turning_angle(0, 1);
            REQUIRE(split_pts.size() == 1);
            const auto turning_angle_1 = curve.get_turning_angle(0, split_pts[0]);
            const auto turning_angle_2 = curve.get_turning_angle(split_pts[0], 1);
            REQUIRE(std::abs(turning_angle_1) == Approx(M_PI/4));
            REQUIRE(std::abs(turning_angle_2) == Approx(M_PI/4));
#endif
        }

        SECTION("Singularity") {
#if NANOSPLINE_SYMPY
            auto singular_pts = curve.compute_singularities(0, 1);
            REQUIRE(singular_pts.size() == 0);
#endif
        }

        SECTION("Curve with singularity") {
#if NANOSPLINE_SYMPY
            control_pts << 0.0, 0.0,
                           1.0, 1.0,
                           0.0, 1.0,
                           1.0, 0.0;
            curve.set_control_points(control_pts);
            const auto total_turning_angle = curve.get_turning_angle(0, 1);
            REQUIRE(total_turning_angle == Approx(1.5 * M_PI));
            const auto split_pts = curve.reduce_turning_angle(0, 1);
            REQUIRE(split_pts.size() == 1);
            const auto turning_angle_1 = curve.get_turning_angle(0, split_pts[0]);
            const auto turning_angle_2 = curve.get_turning_angle(split_pts[0], 1);
            REQUIRE(std::abs(turning_angle_1) == Approx(0.25 * M_PI));
            REQUIRE(std::abs(turning_angle_2) == Approx(0.25 * M_PI));

            auto singular_pts = curve.compute_singularities(0, 1);
            REQUIRE(singular_pts.size() == 1);
            REQUIRE(singular_pts[0] == Approx(0.5));
            REQUIRE(curve.evaluate_derivative(0.5).norm() == Approx(0.0));
#endif
        }

        SECTION("degree elevation") {
            auto new_curve = curve.elevate_degree();
            REQUIRE(new_curve.get_degree() == 4);
            assert_same(curve, new_curve, 10);
        }

        SECTION("Update") {
            offset_and_validate(curve);
        }
    }

    SECTION("Dynmaic degree") {
        Eigen::Matrix<Scalar, 4, 2> control_pts;
        control_pts << 0.0, 0.0,
                       1.0, 1.0,
                       2.0, 1.0,
                       3.0, 0.0;
        Bezier<Scalar, 2, -1> curve;
        curve.set_control_points(control_pts);

        SECTION("Ends") {
            auto start = curve.evaluate(0);
            auto end = curve.evaluate(1);

            REQUIRE((start-control_pts.row(0)).norm() == Approx(0.0));
            REQUIRE((end-control_pts.row(3)).norm() == Approx(0.0));
        }

        SECTION("Mid point") {
            auto p = curve.evaluate(0.5);
            REQUIRE(p[0] == Approx(1.5));
            REQUIRE(p[1] > 0.0);
            REQUIRE(p[1] < 1.0);
        }

        SECTION("Inverse evaluation") {
            Eigen::Matrix<Scalar, 1, 2> p(0.0, 1.0);
            REQUIRE_THROWS(curve.inverse_evaluate(p));
        }

        SECTION("Approximate inverse evaluation") {
            validate_approximate_inverse_evaluation(curve, 10);
        }

        SECTION("Derivative") {
            validate_derivatives(curve, 10);
            validate_2nd_derivatives(curve, 10);
        }

        SECTION("Turning angle") {
#if NANOSPLINE_SYMPY
            const auto total_turning_angle = curve.get_turning_angle(0, 1);
            REQUIRE(std::abs(total_turning_angle) == Approx(M_PI/2));
            const auto split_pts = curve.reduce_turning_angle(0, 1);
            REQUIRE(split_pts.size() == 1);
            const auto turning_angle_1 = curve.get_turning_angle(0, split_pts[0]);
            const auto turning_angle_2 = curve.get_turning_angle(split_pts[0], 1);
            REQUIRE(std::abs(turning_angle_1) == Approx(M_PI/4));
            REQUIRE(std::abs(turning_angle_2) == Approx(M_PI/4));
#endif
        }

        SECTION("Turning angle of linear curve") {
#if NANOSPLINE_SYMPY
            control_pts.col(1).setZero();
            curve.set_control_points(control_pts);
            const auto total_turning_angle = curve.get_turning_angle(0, 1);
            REQUIRE(total_turning_angle == Approx(0.0));
            const auto split_pts = curve.reduce_turning_angle(0, 1);
            REQUIRE(split_pts.empty());
#endif
        }

        SECTION("Singularity") {
#if NANOSPLINE_SYMPY
            auto singular_pts = curve.compute_singularities(0, 1);
            REQUIRE(singular_pts.size() == 0);
#endif
        }

        SECTION("degree elevation") {
            auto new_curve = curve.elevate_degree();
            REQUIRE(new_curve.get_degree() == 4);
            assert_same(curve, new_curve, 10);
        }

        SECTION("Update") {
            offset_and_validate(curve);
        }
    }

    SECTION("Specialized degree 0") {
        Eigen::Matrix<Scalar, 1, 2> control_pts;
        control_pts << 0.0, 0.1;
        Bezier<Scalar, 2, 0> curve;
        curve.set_control_points(control_pts);

        Eigen::Matrix<Scalar, 1, 2> p(0.0, 1.0);
        curve.inverse_evaluate(p);

        SECTION("Consistency") {
            Bezier<Scalar, 2, 0, true> generic_curve;
            generic_curve.set_control_points(control_pts);
            assert_same(curve, generic_curve, 10);
        }

        SECTION("Derivative") {
            validate_derivatives(curve, 10);
            validate_2nd_derivatives(curve, 10);
        }

        SECTION("degree elevation") {
            auto new_curve = curve.elevate_degree();
            REQUIRE(new_curve.get_degree() == 1);
            assert_same(curve, new_curve, 10);
        }

        //SECTION("Turning angle") {
        //    const auto total_turning_angle = curve.get_turning_angle(0, 1);
        //    REQUIRE(total_turning_angle == Approx(0.0));
        //    const auto split_pts = curve.reduce_turning_angle(0, 1);
        //    REQUIRE(split_pts.size() == 0);
        //}

        SECTION("Update") {
            offset_and_validate(curve);
        }
    }

    SECTION("Specialized degree 1") {
        Eigen::Matrix<Scalar, 2, 2> control_pts;
        control_pts << 0.0, 0.0,
                       1.0, 1.0;
        Bezier<Scalar, 2, 1> curve;
        curve.set_control_points(control_pts);

        SECTION("Consistency") {
            Bezier<Scalar, 2, 1, true> generic_curve;
            generic_curve.set_control_points(control_pts);
            assert_same(curve, generic_curve, 10);
        }

        SECTION("Evaluation") {
            auto start = curve.evaluate(0.0);
            auto mid = curve.evaluate(0.5);
            auto end = curve.evaluate(1.0);

            REQUIRE((start - control_pts.row(0)).norm() == Approx(0.0));
            REQUIRE((end - control_pts.row(1)).norm() == Approx(0.0));
            REQUIRE(mid[0] == Approx(0.5));
            REQUIRE(mid[1] == Approx(0.5));
        }

        SECTION("Inverse evaluate") {
            Scalar t0 = 0.2f;
            const auto p0 = curve.evaluate(t0);
            const auto t = curve.inverse_evaluate(p0);
            REQUIRE(t0 == Approx(t));

            Eigen::Matrix<Scalar, 1, 2> p1(1.0, 0.0);
            const auto t1 = curve.inverse_evaluate(p1);
            REQUIRE(t1 == Approx(0.5));

            Eigen::Matrix<Scalar, 1, 2> p2(-1.0, 0.0);
            const auto t2 = curve.inverse_evaluate(p2);
            REQUIRE(t2 == Approx(0.0));

            Eigen::Matrix<Scalar, 1, 2> p3(1.0, 1.1);
            const auto t3 = curve.inverse_evaluate(p3);
            REQUIRE(t3 == Approx(1.0));
        }

        SECTION("Derivative") {
            validate_derivatives(curve, 10);
            validate_2nd_derivatives(curve, 10);
        }

        SECTION("Turning angle") {
#if NANOSPLINE_SYMPY
            const auto total_turning_angle = curve.get_turning_angle(0, 1);
            REQUIRE(total_turning_angle == Approx(0.0).margin(1e-6));
            const auto split_pts = curve.reduce_turning_angle(0, 1);
            REQUIRE(split_pts.size() == 0);
#endif
        }

        SECTION("degree elevation") {
            auto new_curve = curve.elevate_degree();
            REQUIRE(new_curve.get_degree() == 2);
            assert_same(curve, new_curve, 10);
        }

        SECTION("Update") {
            offset_and_validate(curve);
        }
    }

    SECTION("Specialized degree 2") {
        Eigen::Matrix<Scalar, 3, 2> control_pts;
        control_pts << 0.0, 0.0,
                       1.0, 1.0,
                       2.0, 0.0;
        Bezier<Scalar, 2, 2> curve;
        curve.set_control_points(control_pts);

        SECTION("Consistency") {
            Bezier<Scalar, 2, 2, true> generic_curve;
            generic_curve.set_control_points(control_pts);
            assert_same(curve, generic_curve, 10);
        }

        SECTION("Evaluation") {
            const auto start = curve.evaluate(0.0);
            const auto mid = curve.evaluate(0.5);
            const auto end = curve.evaluate(1.0);

            REQUIRE((start-control_pts.row(0)).norm() == Approx(0.0));
            REQUIRE((end-control_pts.row(2)).norm() == Approx(0.0));
            REQUIRE(mid[0] == Approx(1.0));
            REQUIRE(mid[1] < 1.0);
        }

        SECTION("Derivative") {
            validate_derivatives(curve, 10);
            validate_2nd_derivatives(curve, 10);
        }

        SECTION("Turning angle") {
#if NANOSPLINE_SYMPY
            const auto total_turning_angle = curve.get_turning_angle(0, 1);
            REQUIRE(std::abs(total_turning_angle) == Approx(M_PI/2).margin(1e-6));
            const auto split_pts = curve.reduce_turning_angle(0, 1);
            REQUIRE(split_pts.size() == 1);
            REQUIRE(split_pts[0] == Approx(0.5));
            const auto turning_angle_1 = curve.get_turning_angle(0, split_pts[0]);
            const auto turning_angle_2 = curve.get_turning_angle(split_pts[0], 1);
            REQUIRE(std::abs(turning_angle_1) == Approx(M_PI/4));
            REQUIRE(std::abs(turning_angle_2) == Approx(M_PI/4));
#endif
        }

        SECTION("Singularity") {
#if NANOSPLINE_SYMPY
            auto singular_pts = curve.compute_singularities(0, 1);
            REQUIRE(singular_pts.size() == 0);

            control_pts(2, 0) = 0;
            curve.set_control_points(control_pts);
            singular_pts = curve.compute_singularities(0, 1);
            REQUIRE(singular_pts.size() == 1);
            REQUIRE(singular_pts[0] == Approx(0.5));
            REQUIRE(curve.evaluate_derivative(0.5).norm() == Approx(0.0));
#endif
        }

        SECTION("degree elevation") {
            auto new_curve = curve.elevate_degree();
            REQUIRE(new_curve.get_degree() == 3);
            assert_same(curve, new_curve, 10);
        }

        SECTION("Approximate inverse evaluation") {
            validate_approximate_inverse_evaluation(curve, 10);
        }

        SECTION("Update") {
            offset_and_validate(curve);
        }
    }

    SECTION("Specialized degree 3") {
        Eigen::Matrix<Scalar, 4, 2> control_pts;
        control_pts << 0.0, 0.0,
                       1.0, 1.0,
                       2.0, 1.0,
                       3.0, 0.0;
        Bezier<Scalar, 2, 3> curve;
        curve.set_control_points(control_pts);

        SECTION("Consistency") {
            Bezier<Scalar, 2, 3, true> generic_curve;
            generic_curve.set_control_points(control_pts);
            assert_same(curve, generic_curve, 10);
        }

        SECTION("Evaluation") {
            const auto start = curve.evaluate(0.0);
            const auto mid = curve.evaluate(0.5);
            const auto end = curve.evaluate(1.0);

            REQUIRE((start-control_pts.row(0)).norm() == Approx(0.0));
            REQUIRE((end-control_pts.row(3)).norm() == Approx(0.0));
            REQUIRE(mid[0] == Approx(1.5));
            REQUIRE(mid[1] < 1.0);
        }

        SECTION("Derivative") {
            validate_derivatives(curve, 10);
            validate_2nd_derivatives(curve, 10);
        }

        SECTION("Turning angle") {
#if NANOSPLINE_SYMPY
            const auto total_turning_angle = curve.get_turning_angle(0, 1);
            REQUIRE(std::abs(total_turning_angle) == Approx(M_PI/2).margin(1e-6));
            const auto split_pts = curve.reduce_turning_angle(0, 1);
            REQUIRE(split_pts.size() == 1);
            REQUIRE(split_pts[0] == Approx(0.5));
            const auto turning_angle_1 = curve.get_turning_angle(0, split_pts[0]);
            const auto turning_angle_2 = curve.get_turning_angle(split_pts[0], 1);
            REQUIRE(std::abs(turning_angle_1) == Approx(M_PI/4));
            REQUIRE(std::abs(turning_angle_2) == Approx(M_PI/4));
#endif
        }

        SECTION("Turning angle of linear curve") {
#if NANOSPLINE_SYMPY
            control_pts.col(1).setZero();
            curve.set_control_points(control_pts);
            const auto total_turning_angle = curve.get_turning_angle(0, 1);
            REQUIRE(total_turning_angle == Approx(0.0));
            const auto split_pts = curve.reduce_turning_angle(0, 1);
            REQUIRE(split_pts.empty());
#endif
        }

        SECTION("Singularity") {
#if NANOSPLINE_SYMPY
            auto singular_pts = curve.compute_singularities(0, 1);
            REQUIRE(singular_pts.size() == 0);
#endif
        }

        SECTION("Curve with singularity") {
#if NANOSPLINE_SYMPY
            control_pts << 0.0, 0.0,
                           1.0, 1.0,
                           0.0, 1.0,
                           1.0, 0.0;
            curve.set_control_points(control_pts);
            const auto total_turning_angle = curve.get_turning_angle(0, 1);
            REQUIRE(total_turning_angle == Approx(1.5 * M_PI));
            const auto split_pts = curve.reduce_turning_angle(0, 1);
            REQUIRE(split_pts.size() == 1);
            const auto turning_angle_1 = curve.get_turning_angle(0, split_pts[0]);
            const auto turning_angle_2 = curve.get_turning_angle(split_pts[0], 1);
            REQUIRE(std::abs(turning_angle_1) == Approx(0.25 * M_PI));
            REQUIRE(std::abs(turning_angle_2) == Approx(0.25 * M_PI));

            auto singular_pts = curve.compute_singularities(0, 1);
            REQUIRE(singular_pts.size() == 1);
            REQUIRE(singular_pts[0] == Approx(0.5));
            REQUIRE(curve.evaluate_derivative(0.5).norm() == Approx(0.0));
#endif
        }

        SECTION("degree elevation") {
            auto new_curve = curve.elevate_degree();
            REQUIRE(new_curve.get_degree() == 4);
            assert_same(curve, new_curve, 10);
        }

        SECTION("Update") {
            offset_and_validate(curve);
        }
    }

    SECTION("Extrapolation") {
        Eigen::Matrix<Scalar, 4, 2> control_pts;
        control_pts << 0.0, 0.0,
                       1.0, 1.0,
                       1.0, 0.0,
                       0.0, 1.0;
        Bezier<Scalar, 2, -1> curve;
        curve.set_control_points(control_pts);

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
        REQUIRE(dd0[0] == Approx(dd1[0]));
        REQUIRE(dd0[1] == Approx(-dd1[1]));
    }

    SECTION("Periodic curve") {
        Eigen::Matrix<Scalar, 4, 2> control_pts;
        control_pts << 0.0, 0.0,
                       1.0, 0.0,
                       0.0, 1.0,
                       0.0, 0.0;
        Bezier<Scalar, 2> curve;
        curve.set_control_points(control_pts);

        REQUIRE(curve.is_closed());
        REQUIRE(!curve.is_closed(1));
        REQUIRE(!curve.is_closed(2));

        curve.set_periodic(true);
        assert_same(curve, curve, 10, 0.1, 0.9, 1.1, 1.9);
        assert_same(curve, curve, 10, 0.9, 1.1, 1.9, 2.1);

        Eigen::Matrix<Scalar, 1, 2> q(-1, -1);
        Scalar t0 = curve.approximate_inverse_evaluate(q, 0, 1);
        REQUIRE(curve.evaluate(t0).norm() == Approx(0.0));
        Scalar t1 = curve.approximate_inverse_evaluate(q, 1, 2);
        REQUIRE(curve.evaluate(t1).norm() == Approx(0.0));
        Scalar t2 = curve.approximate_inverse_evaluate(q, 2.5, 3.5, 10);
        REQUIRE(curve.evaluate(t2).norm() == Approx(0.0).margin(1e-3));
        Scalar t3 = curve.approximate_inverse_evaluate(q, -2.1, -1.9, 10);
        REQUIRE(curve.evaluate(t3).norm() == Approx(0.0).margin(1e-3));
    }
}
