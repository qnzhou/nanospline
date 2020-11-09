#include <catch2/catch.hpp>
#include <iostream>

#include <nanospline/BSplinePatch.h>
#include <nanospline/forward_declaration.h>

#include "validation_utils.h"

TEST_CASE("BSplinePatch", "[nonrational][bspline_patch]") {
    using namespace nanospline;
    using Scalar = double;

    SECTION("Bilinear patch non-planar") {
        BSplinePatch<Scalar, 3, 1, 1> patch;
        Eigen::Matrix<Scalar, 4, 3> control_grid;
        control_grid <<
            0.0, 0.0, 0.0,
            0.0, 1.0, 1.0,
            1.0, 0.0, 1.0,
            1.0, 1.0, 0.0;
        patch.set_control_grid(control_grid);

        Eigen::Matrix<Scalar, 4, 1> knots_u, knots_v;
        knots_u << 0.0, 0.0, 1.0, 1.0;
        knots_v << 0.0, 0.0, 1.0, 1.0;
        patch.set_knots_u(knots_u);
        patch.set_knots_v(knots_v);
        patch.initialize();

        REQUIRE(patch.get_degree_u() == 1);
        REQUIRE(patch.get_degree_v() == 1);

        const auto corner_00 = patch.evaluate(0.0, 0.0);
        const auto corner_01 = patch.evaluate(0.0, 1.0);
        const auto corner_11 = patch.evaluate(1.0, 1.0);
        const auto corner_10 = patch.evaluate(1.0, 0.0);
        REQUIRE((corner_00 - control_grid.row(0)).norm() == Approx(0.0));
        REQUIRE((corner_01 - control_grid.row(1)).norm() == Approx(0.0));
        REQUIRE((corner_10 - control_grid.row(2)).norm() == Approx(0.0));
        REQUIRE((corner_11 - control_grid.row(3)).norm() == Approx(0.0));

        const auto p_mid = patch.evaluate(0.5, 0.5);
        REQUIRE(p_mid[0] == Approx(0.5));
        REQUIRE(p_mid[1] == Approx(0.5));
        REQUIRE(p_mid[2] == Approx(0.5));

        validate_iso_curves(patch, 10);
        validate_derivative(patch, 10, 10);
        validate_derivative_patches(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10);
        validate_inverse_evaluation_3d(patch, 5, 5);
    }

    SECTION("Cubic patch") {
        BSplinePatch<Scalar, 3, 3, 3> patch;
        Eigen::Matrix<Scalar, 16, 3> control_grid;
        for (int i=0; i<4; i++) {
            for (int j=0; j<4; j++) {
                control_grid.row(i*4+j) << j, i, ((i+j)%2==0)?-1:1;
            }
        }
        patch.set_control_grid(control_grid);
        Eigen::Matrix<Scalar, 8, 1> knots_u, knots_v;
        knots_u << 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0;
        knots_v << 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 2.5, 2.5;
        //knots_v << 0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 2.5;
        patch.set_knots_u(knots_u);
        patch.set_knots_v(knots_v);
        patch.initialize();

        validate_iso_curves(patch, 10);
        validate_derivative(patch, 10, 10);
        validate_derivative_patches(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10);
        validate_inverse_evaluation_3d(patch, 5,5);
    }
    SECTION("Cubic spline") {
        BSplinePatch<Scalar, 3, 3, 3> patch;
        Eigen::Matrix<Scalar, 64, 3> control_grid;
        for (int i=0; i<8; i++) {
            for (int j=0; j<8; j++) {
                control_grid.row(i*8+j) << j, i, ((i+j)%2==0)?-1:1;
            }
        }
        patch.set_control_grid(control_grid);
        Eigen::Matrix<Scalar, 12, 1> knots_u, knots_v;
        knots_u << 0.0, 0.0, 0.0, 0.0, .25, .5, .75, .9, 1.0, 1.0, 1.0, 1.0;
        knots_v << 0.0, 0.0, 0.0, 0.0, .25, 1., 1.5, 1.9, 2.5, 2.5, 2.5, 2.5;
        patch.set_knots_u(knots_u);
        patch.set_knots_v(knots_v);
        patch.initialize();

        validate_iso_curves(patch, 10);
        validate_derivative(patch, 10, 10);
        validate_derivative_patches(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10);
        validate_inverse_evaluation_3d(patch, 10,10);
    }
    SECTION("Degree 1 patch") {
        BSplinePatch<Scalar, 3, -1, -1> patch;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 3> control_grid(4, 3);
        control_grid << -3.3, -10.225317547305501, 0.0,
                        -3.3, -10.225317547305501, 0.5,
                        -3.8000000000000003, -10.225317547305501, 0.0,
                        -3.8000000000000003, -10.225317547305501, 0.5;
        patch.set_control_grid(control_grid);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> u_knots(4, 1);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> v_knots(4, 1);
        u_knots << -2.0234899551297305, -2.0234899551297305, -1.52348995512973, -1.52348995512973;
        v_knots << 32.5856708170825, 32.5856708170825, 33.0856708170825, 33.0856708170825;
        patch.set_knots_u(u_knots);
        patch.set_knots_v(v_knots);
        patch.set_degree_u(1);
        patch.set_degree_v(1);
        patch.initialize();

        REQUIRE(patch.get_degree_u() == 1);
        REQUIRE(patch.get_degree_v() == 1);
        validate_iso_curves(patch, 10);
        validate_derivative(patch, 10, 10);
        validate_derivative_patches(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10);
        validate_inverse_evaluation_3d(patch, 5, 5);
    }

    SECTION("Mixed degree") {
        BSplinePatch<Scalar, 3, -1, -1> patch;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 3> control_grid(14, 3);
        control_grid << 31.75, 0.0, 12.700000000000001,
                        31.75, 0.0, 0.0,
                        31.75, -54.99261314031184, 12.700000000000001,
                        31.75, -54.99261314031184, 0.0,
                        -15.874999999999993, -27.49630657015593, 12.700000000000001,
                        -15.874999999999993, -27.49630657015593, 0.0,
                        -63.499999999999986, -7.776507174585691e-15, 12.700000000000001,
                        -63.499999999999986, -7.776507174585691e-15, 0.0,
                        -15.875000000000014, 27.49630657015592, 12.700000000000001,
                        -15.875000000000014, 27.49630657015592, 0.0,
                        31.74999999999995, 54.99261314031187, 12.700000000000001,
                        31.74999999999995, 54.99261314031187, 0.0,
                        31.75, 7.776507174585693e-15, 12.700000000000001,
                        31.75, 7.776507174585693e-15, 0.0;
        patch.set_control_grid(control_grid);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> u_knots(10, 1);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> v_knots(4, 1);
        u_knots << 0.0, 
                   0.0, 
                   0.0, 
                   2.0943951023931953, 
                   2.0943951023931953, 
                   4.1887902047863905, 
                   4.1887902047863905, 
                   6.283185307179586, 
                   6.283185307179586, 
                   6.283185307179586;
        v_knots << -10.573884999451131, 
                   -10.573884999451131, 
                   2.12611500054887, 
                   2.12611500054887;
        patch.set_knots_u(u_knots);
        patch.set_knots_v(v_knots);
        patch.set_degree_u(2);
        patch.set_degree_v(1);
        patch.initialize();

        REQUIRE(patch.get_degree_u() == 2);
        REQUIRE(patch.get_degree_v() == 1);
        validate_iso_curves(patch, 10);
        validate_derivative(patch, 10, 10);
        validate_derivative_patches(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10);
    }

    SECTION("Debug example") {
        BSplinePatch<Scalar, 3, -1, -1> patch;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 3> control_grid(6, 3);
        control_grid <<
            326.0, 1385.0, 19.999999999999996,
            326.0, 1385.0, 36.0,
            351.0, 1385.0, 19.999999999999996,
            351.0, 1385.0, 36.0,
            351.0, 1410.0, 19.999999999999996,
            351.0, 1410.0, 36.0;
        patch.set_control_grid(control_grid);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> u_knots(6, 1);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> v_knots(4, 1);
        u_knots << 
            1.5707963267948966,
            1.5707963267948966,
            1.5707963267948966,
            3.141592653589793,
            3.141592653589793,
            3.141592653589793;
        v_knots <<
            -16.000000000000004,
            -16.000000000000004,
            0.0,
            0.0;
        patch.set_knots_u(u_knots);
        patch.set_knots_v(v_knots);
        patch.set_degree_u(2);
        patch.set_degree_v(1);
        patch.initialize();

        REQUIRE(patch.get_degree_u() == 2);
        REQUIRE(patch.get_degree_v() == 1);
        validate_iso_curves(patch, 10);
        validate_derivative(patch, 10, 10);
        validate_derivative_patches(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10);

        // Out of bound extrapolation.
        const auto p0 = patch.evaluate(1.5707963267948966, -16.000000000000011);
        const auto p1 = patch.evaluate(1.5707963267948966, -16.000000000000004);
        REQUIRE((p0-p1).norm() == Approx(0.0).margin(1e-12));
    }

    SECTION("Extrapolation") {
        BSplinePatch<Scalar, 3, 3, 3> patch;
        Eigen::Matrix<Scalar, 16, 3> control_grid;
        for (int i=0; i<4; i++) {
            for (int j=0; j<4; j++) {
                control_grid.row(i*4+j) << j, i, ((i+j)%2==0)?-1:1;
            }
        }
        patch.set_control_grid(control_grid);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> u_knots(8, 1);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> v_knots(8, 1);
        u_knots << 0, 0, 0, 0, 1, 1, 1, 1;
        v_knots << 0, 0, 0, 0, 2, 2, 2, 2;
        patch.set_knots_u(u_knots);
        patch.set_knots_v(v_knots);
        patch.initialize();

        constexpr Scalar d = 0.1;
        const auto u_min = patch.get_u_lower_bound();
        const auto u_max = patch.get_u_upper_bound();
        const auto v_min = patch.get_v_lower_bound();
        const auto v_max = patch.get_v_upper_bound();

        const auto corner_00 = patch.evaluate(u_min - d, v_min - d);
        const auto corner_01 = patch.evaluate(u_max + d, v_min - d);
        const auto corner_11 = patch.evaluate(u_max + d, v_max + d);
        const auto corner_10 = patch.evaluate(u_min - d, v_max + d);

        REQUIRE(corner_00[0] < 0);
        REQUIRE(corner_00[1] < 0);
        REQUIRE(corner_10[0] > 3);
        REQUIRE(corner_10[1] < 0);
        REQUIRE(corner_11[0] > 3);
        REQUIRE(corner_11[1] > 3);
        REQUIRE(corner_01[0] < 0);
        REQUIRE(corner_01[1] > 3);
    }
}

TEST_CASE("BSplinePatch Benchmark", "[!benchmark][bspline_patch]") {
    using namespace nanospline;
    using Scalar = double;

    BSplinePatch<Scalar, 3, 3, 3> patch;
    Eigen::Matrix<Scalar, 16, 3> control_grid;
    for (int i=0; i<4; i++) {
        for (int j=0; j<4; j++) {
            control_grid.row(i*4+j) << j, i, ((i+j)%2==0)?-1:1;
        }
    }
    patch.set_control_grid(control_grid);
    Eigen::Matrix<Scalar, 8, 1> knots_u, knots_v;
    knots_u << 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0;
    knots_v << 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 2.5, 2.5;
    patch.set_knots_u(knots_u);
    patch.set_knots_v(knots_v);
    patch.initialize();

    BENCHMARK("Evaluation") {
        return patch.evaluate(0.5, 0.6);
    };

    BENCHMARK("Derivative") {
        auto du = patch.evaluate_derivative_u(0.5, 0.6);
        auto dv = patch.evaluate_derivative_u(0.5, 0.6);
        Eigen::Matrix<Scalar, 2, 3> grad;
        grad << du, dv;
        return grad;
    };

    BENCHMARK("2nd Derivative") {
        Eigen::Matrix<Scalar, 3, 3> hessian;
        hessian.row(0) = patch.evaluate_2nd_derivative_uu(0.5, 0.6);
        hessian.row(1) = patch.evaluate_2nd_derivative_vv(0.5, 0.6);
        hessian.row(2) = patch.evaluate_2nd_derivative_uv(0.5, 0.6);
        return hessian;
    };
}

