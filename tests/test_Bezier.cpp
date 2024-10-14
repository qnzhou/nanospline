#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

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

        REQUIRE_THAT((start-control_pts.row(0)).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
        REQUIRE_THAT((end-control_pts.row(0)).norm(),   Catch::Matchers::WithinAbs(0.0, 1e-6));
        REQUIRE_THAT((mid-control_pts.row(0)).norm(),   Catch::Matchers::WithinAbs(0.0, 1e-6));

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

        REQUIRE_THAT(start[0], Catch::Matchers::WithinAbs(0.0, 1e-6));
        REQUIRE_THAT(mid[0], Catch::Matchers::WithinAbs(0.5, 1e-6));
        REQUIRE_THAT(end[0], Catch::Matchers::WithinAbs(1.0, 1e-6));

        REQUIRE_THAT(start[1], Catch::Matchers::WithinAbs(0.0, 1e-6));
        REQUIRE_THAT(mid[1], Catch::Matchers::WithinAbs(0.0, 1e-6));
        REQUIRE_THAT(end[1], Catch::Matchers::WithinAbs(0.0, 1e-6));

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

            REQUIRE_THAT((start-control_pts.row(0)).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
            REQUIRE_THAT((end-control_pts.row(3)).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
        }

        SECTION("Mid point") {
            auto p = curve.evaluate(0.5);
            REQUIRE_THAT(p[0], Catch::Matchers::WithinAbs(1.5, 1e-6));
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
            REQUIRE_THAT(total_turning_angle, Catch::Matchers::WithinAbs(M_PI/3, 1e-6));
            const auto split_pts = curve.reduce_turning_angle(0, 1);
            REQUIRE(split_pts.size() == 1);
            const auto turning_angle_1 = curve.get_turning_angle(0, split_pts[0]);
            const auto turning_angle_2 = curve.get_turning_angle(split_pts[0], 1);
            REQUIRE_THAT(turning_angle_1, Catch::Matchers::WithinAbs(M_PI/4, 1e-6));
            REQUIRE_THAT(turning_angle_2, Catch::Matchers::WithinAbs(M_PI/4, 1e-6));
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
            REQUIRE(total_turning_angle == Catch::Matchers::WithinAbs(1.5 * M_PI, 1e-6));
            const auto split_pts = curve.reduce_turning_angle(0, 1);
            REQUIRE(split_pts.size() == 1);
            const auto turning_angle_1 = curve.get_turning_angle(0, split_pts[0]);
            const auto turning_angle_2 = curve.get_turning_angle(split_pts[0], 1);
            REQUIRE(std::abs(turning_angle_1) == Catch::Matchers::WithinAbs(0.25 * M_PI, 1e-6));
            REQUIRE(std::abs(turning_angle_2) == Catch::Matchers::WithinAbs(0.25 * M_PI, 1e-6));

            auto singular_pts = curve.compute_singularities(0, 1);
            REQUIRE(singular_pts.size() == 1);
            REQUIRE(singular_pts[0] == Catch::Matchers::WithinAbs(0.5, 1e-6));
            REQUIRE(curve.evaluate_derivative(0.5).norm() == Catch::Matchers::WithinAbs(0.0, 1e-6));
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

            REQUIRE_THAT((start-control_pts.row(0)).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
            REQUIRE_THAT((end-control_pts.row(3)).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
        }

        SECTION("Mid point") {
            auto p = curve.evaluate(0.5);
            REQUIRE_THAT(p[0], Catch::Matchers::WithinAbs(1.5, 1e-6));
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
            REQUIRE_THAT(std::abs(total_turning_angle), Catch::Matchers::WithinAbs(M_PI/2, 1e-6));
            const auto split_pts = curve.reduce_turning_angle(0, 1);
            REQUIRE(split_pts.size() == 1);
            const auto turning_angle_1 = curve.get_turning_angle(0, split_pts[0]);
            const auto turning_angle_2 = curve.get_turning_angle(split_pts[0], 1);
            REQUIRE_THAT(std::abs(turning_angle_1), Catch::Matchers::WithinAbs(M_PI/4, 1e-6));
            REQUIRE_THAT(std::abs(turning_angle_2), Catch::Matchers::WithinAbs(M_PI/4, 1e-6));
#endif
        }

        SECTION("Turning angle of linear curve") {
#if NANOSPLINE_SYMPY
            control_pts.col(1).setZero();
            curve.set_control_points(control_pts);
            const auto total_turning_angle = curve.get_turning_angle(0, 1);
            REQUIRE_THAT(total_turning_angle, Catch::Matchers::WithinAbs(0.0, 1e-6));
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
        //    REQUIRE_THAT(total_turning_angle, Catch::Matchers::WithinAbs(0.0, 1e-6));
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

            REQUIRE_THAT((start - control_pts.row(0)).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
            REQUIRE_THAT((end - control_pts.row(1)).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
            REQUIRE_THAT(mid[0], Catch::Matchers::WithinAbs(0.5, 1e-6));
            REQUIRE_THAT(mid[1], Catch::Matchers::WithinAbs(0.5, 1e-6));
        }

        SECTION("Inverse evaluate") {
            Scalar t0 = 0.2f;
            const auto p0 = curve.evaluate(t0);
            const auto t = curve.inverse_evaluate(p0);
            REQUIRE_THAT(t0, Catch::Matchers::WithinAbs(t, 1e-6));

            Eigen::Matrix<Scalar, 1, 2> p1(1.0, 0.0);
            const auto t1 = curve.inverse_evaluate(p1);
            REQUIRE_THAT(t1, Catch::Matchers::WithinAbs(0.5, 1e-6));

            Eigen::Matrix<Scalar, 1, 2> p2(-1.0, 0.0);
            const auto t2 = curve.inverse_evaluate(p2);
            REQUIRE_THAT(t2, Catch::Matchers::WithinAbs(0.0, 1e-6));

            Eigen::Matrix<Scalar, 1, 2> p3(1.0, 1.1);
            const auto t3 = curve.inverse_evaluate(p3);
            REQUIRE_THAT(t3, Catch::Matchers::WithinAbs(1.0, 1e-6));
        }

        SECTION("Derivative") {
            validate_derivatives(curve, 10);
            validate_2nd_derivatives(curve, 10);
        }

        SECTION("Turning angle") {
#if NANOSPLINE_SYMPY
            const auto total_turning_angle = curve.get_turning_angle(0, 1);
            REQUIRE_THAT(total_turning_angle, Catch::Matchers::WithinAbs(0.0, 1e-6));
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

            REQUIRE_THAT((start - control_pts.row(0)).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
            REQUIRE_THAT((end - control_pts.row(2)).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
            REQUIRE_THAT(mid[0], Catch::Matchers::WithinAbs(1.0, 1e-6));
            REQUIRE(mid[1] < 1.0);
        }

        SECTION("Derivative") {
            validate_derivatives(curve, 10);
            validate_2nd_derivatives(curve, 10);
        }

        SECTION("Turning angle") {
#if NANOSPLINE_SYMPY
            const auto total_turning_angle = curve.get_turning_angle(0, 1);
            REQUIRE_THAT(std::abs(total_turning_angle), Catch::Matchers::WithinAbs(M_PI/2, 1e-6));
            const auto split_pts = curve.reduce_turning_angle(0, 1);
            REQUIRE(split_pts.size() == 1);
            REQUIRE_THAT(split_pts[0], Catch::Matchers::WithinAbs(0.5, 1e-6));
            const auto turning_angle_1 = curve.get_turning_angle(0, split_pts[0]);
            const auto turning_angle_2 = curve.get_turning_angle(split_pts[0], 1);
            REQUIRE_THAT(std::abs(turning_angle_1), Catch::Matchers::WithinAbs(M_PI/4, 1e-6));
            REQUIRE_THAT(std::abs(turning_angle_2), Catch::Matchers::WithinAbs(M_PI/4, 1e-6));
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
            REQUIRE(singular_pts[0] == Catch::Matchers::WithinAbs(0.5, 1e-6));
            REQUIRE(curve.evaluate_derivative(0.5).norm() == Catch::Matchers::WithinAbs(0.0, 1e-6));
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

            REQUIRE_THAT((start - control_pts.row(0)).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
            REQUIRE_THAT((end - control_pts.row(3)).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
            REQUIRE_THAT(mid[0], Catch::Matchers::WithinAbs(1.5, 1e-6));
            REQUIRE(mid[1] < 1.0);
        }

        SECTION("Derivative") {
            validate_derivatives(curve, 10);
            validate_2nd_derivatives(curve, 10);
        }

        SECTION("Turning angle") {
#if NANOSPLINE_SYMPY
            const auto total_turning_angle = curve.get_turning_angle(0, 1);
            REQUIRE_THAT(std::abs(total_turning_angle), Catch::Matchers::WithinAbs(M_PI/2, 1e-6));
            const auto split_pts = curve.reduce_turning_angle(0, 1);
            REQUIRE(split_pts.size() == 1);
            REQUIRE_THAT(split_pts[0], Catch::Matchers::WithinAbs(0.5, 1e-6));
            const auto turning_angle_1 = curve.get_turning_angle(0, split_pts[0]);
            const auto turning_angle_2 = curve.get_turning_angle(split_pts[0], 1);
            REQUIRE_THAT(std::abs(turning_angle_1), Catch::Matchers::WithinAbs(M_PI/4, 1e-6));
            REQUIRE_THAT(std::abs(turning_angle_2), Catch::Matchers::WithinAbs(M_PI/4, 1e-6));
#endif
        }

        SECTION("Turning angle of linear curve") {
#if NANOSPLINE_SYMPY
            control_pts.col(1).setZero();
            curve.set_control_points(control_pts);
            const auto total_turning_angle = curve.get_turning_angle(0, 1);
            REQUIRE_THAT(total_turning_angle, Catch::Matchers::WithinAbs(0.0, 1e-6));
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
            REQUIRE_THAT(total_turning_angle, Catch::Matchers::WithinAbs(1.5 * M_PI, 1e-6));
            const auto split_pts = curve.reduce_turning_angle(0, 1);
            REQUIRE(split_pts.size() == 1);
            const auto turning_angle_1 = curve.get_turning_angle(0, split_pts[0]);
            const auto turning_angle_2 = curve.get_turning_angle(split_pts[0], 1);
            REQUIRE_THAT(turning_angle_1, Catch::Matchers::WithinAbs(0.25 * M_PI, 1e-6));
            REQUIRE_THAT(turning_angle_2, Catch::Matchers::WithinAbs(0.25 * M_PI, 1e-6));

            auto singular_pts = curve.compute_singularities(0, 1);
            REQUIRE(singular_pts.size() == 1);
            REQUIRE_THAT(singular_pts[0], Catch::Matchers::WithinAbs(0.5, 1e-6));
            REQUIRE_THAT(curve.evaluate_derivative(0.5).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
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
        REQUIRE_THAT(dd0[0], Catch::Matchers::WithinAbs(dd1[0], 1e-6));
        REQUIRE_THAT(dd0[1], Catch::Matchers::WithinAbs(-dd1[1], 1e-6));
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
        REQUIRE_THAT(curve.evaluate(t0).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
        Scalar t1 = curve.approximate_inverse_evaluate(q, 1, 2);
        REQUIRE_THAT(curve.evaluate(t1).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
        Scalar t2 = curve.approximate_inverse_evaluate(q, 2.5, 3.5, 10);
        REQUIRE_THAT(curve.evaluate(t2).norm(), Catch::Matchers::WithinAbs(0.0, 1e-3));
        Scalar t3 = curve.approximate_inverse_evaluate(q, -2.1, -1.9, 10);
        REQUIRE_THAT(curve.evaluate(t3).norm(), Catch::Matchers::WithinAbs(0.0, 1e-3));

        auto curve2 = curve.elevate_degree();
        REQUIRE(curve2.get_periodic());
        assert_same(curve, curve2, 10);
    }
}
