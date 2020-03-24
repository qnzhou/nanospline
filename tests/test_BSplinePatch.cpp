#include <catch2/catch.hpp>
#include <iostream>

#include <nanospline/BSplinePatch.h>
#include <nanospline/save_obj.h>
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
        knots_v << 0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 2.5;
        patch.set_knots_u(knots_u);
        patch.set_knots_v(knots_v);
        patch.initialize();

        validate_iso_curves(patch, 10);
        validate_derivative(patch, 10, 10);
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
    }
}

