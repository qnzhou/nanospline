#include <catch2/catch.hpp>

#include <nanospline/split.h>
#include <nanospline/BezierPatch.h>
#include <nanospline/RationalBezierPatch.h>
#include <nanospline/NURBSPatch.h>
#include <nanospline/BSplinePatch.h>
#include <nanospline/save_svg.h>
#include <nanospline/forward_declaration.h>
#include "validation_utils.h"

template<typename PatchType>
void validate_bezier_patch_splitting(PatchType patch){
  using Scalar = typename PatchType::Scalar;
    SECTION("Degenerate split: u,v=(0,0)") {
        Scalar u = 0.;
        Scalar v = 0.;
        const auto subpatches = patch.split(u, v);
        assert_same(patch, subpatches[0], 10, 
                0., 1., 0., 1., 
                0., 1., 0., 1.);
    }

    SECTION("Degenerate split: u,v=(1,1)") {
        Scalar u = 1.;
        Scalar v = 1.;
        const auto subpatches = patch.split(u, v);
        assert_same(patch, subpatches[0], 10, 
                0., 1., 0., 1., 
                0., 1., 0., 1.);
    }
    SECTION("Degenerate split: u,v=(0,1)") {
        Scalar u = 0.;
        Scalar v = 1.;
        const auto subpatches = patch.split(u, v);
        assert_same(patch, subpatches[0], 10, 
                0., 1., 0., 1., 
                0., 1., 0., 1.);
    }
    SECTION("Degenerate split: u,v=(1,0)") {
        Scalar u = 1.;
        Scalar v = 0.;
        const auto subpatches = patch.split(u, v);
        assert_same(patch, subpatches[0], 10, 
                0., 1., 0., 1., 
                0., 1., 0., 1.);
    }

    SECTION("Split: u=.5") {
        Scalar u = .5;
        const auto subpatches = patch.split_u(u);
        assert_same(patch, subpatches[0], 10, 
                0., u, 0., 1.,
                0., 1., 0., 1.);
        assert_same(patch, subpatches[1], 10, 
                u, 1., 0., 1.,
                0., 1., 0., 1.);
    }
    SECTION("Split: v=.5") {
        Scalar v = .5;
        const auto subpatches = patch.split_v(v);
        assert_same(patch, subpatches[0], 10, 
                0., 1., 0., v,
                0., 1., 0., 1.);
        assert_same(patch, subpatches[1], 10, 
                0., 1., v,  1.,
                0., 1., 0., 1.);
    }
    SECTION("Split u,v: random points:"){
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<Scalar> dist(0, 1);

        int num_split_tests = 30;
        for (int i =0; i < num_split_tests; i++) {
            Scalar u = dist(generator);
            Scalar v = dist(generator);

            const auto subpatches = patch.split(u,v);
            assert_same(patch, subpatches[0], 10, 
                    0., u, 0., v,
                    0., 1., 0., 1.);
            assert_same(patch, subpatches[2], 10, 
                    u,  1., 0., v,
                    0., 1., 0., 1.);
            assert_same(patch, subpatches[1], 10, 
                    0., u,   v, 1.,
                    0., 1., 0., 1.);

            assert_same(patch, subpatches[3], 10, 
                    u,  1.,  v, 1.,
                    0., 1., 0., 1.);
        }
    }
}

template<typename PatchType>
void validate_bspline_patch_splitting(PatchType patch){
  using Scalar = typename PatchType::Scalar;
        Scalar u_min = patch.get_u_lower_bound();
        Scalar u_max = patch.get_u_upper_bound();
        Scalar v_min = patch.get_v_lower_bound();
        Scalar v_max = patch.get_v_upper_bound();
    SECTION("Degenerate split: u,v=(0,0)") {
        Scalar u = u_min;
        Scalar v = v_min;
        const auto subpatches = patch.split(u, v);
        assert_same(patch, subpatches[0], 10, 
                u_min, u_max, v_min, v_max, 
                u_min, u_max, v_min, v_max);
    }

    SECTION("Degenerate split: u,v=(1,1)") {
        Scalar u = u_max;
        Scalar v = v_max;
        const auto subpatches = patch.split(u, v);
        assert_same(patch, subpatches[0], 10, 
                u_min, u_max, v_min, v_max, 
                u_min, u_max, v_min, v_max);
    }
    SECTION("Degenerate split: u,v=(0,1)") {
        Scalar u = u_min;
        Scalar v = v_max;
        const auto subpatches = patch.split(u, v);
        assert_same(patch, subpatches[0], 10, 
                u_min, u_max, v_min, v_max, 
                u_min, u_max, v_min, v_max);
    }
    SECTION("Degenerate split: u,v=(1,0)") {
        Scalar u = u_max;
        Scalar v = v_min;
        const auto subpatches = patch.split(u, v);
        assert_same(patch, subpatches[0], 10, 
                u_min, u_max, v_min, v_max, 
                u_min, u_max, v_min, v_max);
    }

    SECTION("Split: u=midpoint") {
        Scalar u = (u_max - u_min)/2. + u_min;
        const auto subpatches = patch.split_u(u);
        assert_same(patch, subpatches[0], 10, 
                u_min, u, v_min, v_max, 
                u_min, u, v_min, v_max);
        assert_same(patch, subpatches[1], 10, 
                u, u_max, v_min, v_max, 
                u, u_max, v_min, v_max);
    }
    SECTION("Split: v=midpoint") {
        Scalar v = (v_max - v_min)/2. + v_min;
        const auto subpatches = patch.split_v(v);
        assert_same(patch, subpatches[0], 10, 
                u_min, u_max, v_min, v, 
                u_min, u_max, v_min, v);
        assert_same(patch, subpatches[1], 10, 
                u_min, u_max, v, v_max, 
                u_min, u_max, v, v_max);
    }
    SECTION("Split u,v: random points:"){
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<Scalar> dist_u(u_min, u_max);
        std::uniform_real_distribution<Scalar> dist_v(v_min, v_max);

        int num_split_tests = 30;
        for (int i =0; i < num_split_tests; i++) {
            Scalar u = dist_u(generator);
            Scalar v = dist_v(generator);

            const auto subpatches = patch.split(u,v);
            assert_same(patch, subpatches[0], 10, 
                u_min, u, v_min, v, 
                u_min, u, v_min, v);
            
            assert_same(patch, subpatches[2], 10, 
                u, u_max, v_min, v, 
                u, u_max, v_min, v);
            
            assert_same(patch, subpatches[1], 10, 
                u_min, u, v, v_max, 
                u_min, u, v, v_max);

            assert_same(patch, subpatches[3], 10, 
                u, u_max, v, v_max, 
                u, u_max, v, v_max);
        }
    }
}

TEST_CASE("Test curve splitting", "[split][curve]") {
    using namespace nanospline;
    using Scalar = double;

    SECTION("Bezier degree 1") {
        Eigen::Matrix<Scalar, 2, 2> control_pts;
        control_pts << 0.0, 0.0,
                    1.0, 0.0;
        Bezier<Scalar, 2, 1, true> curve;
        curve.set_control_points(control_pts);

        const auto parts = split(curve, 0.5);
        REQUIRE(parts.size() == 2);

        const auto& ctrl_pts_1 = parts[0].get_control_points();
        const auto& ctrl_pts_2 = parts[1].get_control_points();
        REQUIRE((ctrl_pts_1.row(0) - control_pts.row(0)).norm() == Approx(0.0));
        REQUIRE((ctrl_pts_2.row(1) - control_pts.row(1)).norm() == Approx(0.0));
        REQUIRE((ctrl_pts_1.row(1) - ctrl_pts_2.row(0)).norm() == Approx(0.0));
        REQUIRE(ctrl_pts_1(1, 0) == Approx(0.5));
        REQUIRE(ctrl_pts_1(1, 1) == Approx(0.0));
    }

    SECTION("Bezier degree 3") {
        Eigen::Matrix<Scalar, 4, 2> control_pts;
        control_pts << 0.0, 0.0,
                    1.0, 1.0,
                    2.0, 1.0,
                    3.0, 0.0;
        Bezier<Scalar, 2, 3, true> curve;
        curve.set_control_points(control_pts);

        Scalar split_location = 0.0;
        SECTION("Beginning") {
            split_location = 0.0;
        }
        SECTION("Middle") {
            split_location = 0.5;
        }
        SECTION("2/3") {
            split_location = 2.0/3.0;
        }
        SECTION("End") {
            split_location = 1.0;
        }

        const auto parts = split(curve, split_location);
        if(parts.size() == 2){
            assert_same(curve, parts[0], 10, 0.0, split_location, 0.0, 1.0);
            assert_same(curve, parts[1], 10, split_location, 1.0, 0.0, 1.0);
        } else {
            assert_same(curve, parts[0], 10);
        }
    }

    SECTION("Rational bezier degree 1") {
        Eigen::Matrix<Scalar, 2, 2> control_pts;
        control_pts << 0.0, 0.0,
                    1.0, 0.0;
        Eigen::Matrix<Scalar, 2, 1> weights;
        weights << 0.1, 1.0;

        RationalBezier<Scalar, 2, 1, true> curve;
        curve.set_control_points(control_pts);
        curve.set_weights(weights);
        curve.initialize();

        Scalar split_location = 0.0;
        SECTION("Beginning") {
            split_location = 0.0;
        }
        SECTION("Middle") {
            split_location = 0.5;
        }
        SECTION("2/3") {
            split_location = 2.0/3.0;
        }
        SECTION("End") {
            split_location = 1.0;
        }

        const auto parts = split(curve, split_location);
        if(parts.size() == 2){
            assert_same(curve, parts[0], 10, 0.0, split_location, 0.0, 1.0);
            assert_same(curve, parts[1], 10, split_location, 1.0, 0.0, 1.0);
        } else {
            assert_same(curve, parts[0], 10);
        }
    }

    SECTION("Rational bezier degree 3") {
        Eigen::Matrix<Scalar, 4, 2> control_pts;
        control_pts << 0.0, 0.0,
                    1.0, 1.0,
                    2.0,-1.0,
                    3.0, 0.0;
        Eigen::Matrix<Scalar, 4, 1> weights;
        weights << 0.1, 1.0, 0.1, 0.1;

        RationalBezier<Scalar, 2, 3, true> curve;
        curve.set_control_points(control_pts);
        curve.set_weights(weights);
        curve.initialize();

        Scalar split_location = 0.0;
        SECTION("Beginning") {
            split_location = 0.0;
        }
        SECTION("Middle") {
            split_location = 0.5;
        }
        SECTION("2/3") {
            split_location = 2.0/3.0;
        }
        SECTION("End") {
            split_location = 1.0;
        }

        const auto parts = split(curve, split_location);
        if(parts.size() == 2){
            assert_same(curve, parts[0], 10, 0.0, split_location, 0.0, 1.0);
            assert_same(curve, parts[1], 10, split_location, 1.0, 0.0, 1.0);
        } else {
            assert_same(curve, parts[0], 10);
        }
    }

    SECTION("BSpline degree 3") {
        Eigen::Matrix<Scalar, 10, 2> ctrl_pts;
        ctrl_pts << 1, 4, .5, 6, 5, 4, 3, 12, 11, 14, 8, 4, 12, 3, 11, 9, 15, 10, 17, 8;
        Eigen::Matrix<Scalar, 14, 1> knots;
        knots << 0, 0, 0, 0, 1.0/7, 2.0/7, 3.0/7, 4.0/7, 5.0/7, 6.0/7, 1, 1, 1, 1;

        BSpline<Scalar, 2, 3, true> curve;
        curve.set_control_points(ctrl_pts);
        curve.set_knots(knots);

        Scalar split_location = 0.0;
        SECTION("Beginning") {
            split_location = 0.0;
        }
        SECTION("Middle") {
            split_location = 0.5;
        }
        SECTION("2/3") {
            split_location = 2.0/3.0;
        }
        SECTION("End") {
            split_location = 1.0;
        }
        const auto parts = split(curve, split_location);
        if (split_location == 0.0) {
            REQUIRE(parts.size() == 1);
            assert_same(curve, parts[0], 10);
        } else if (split_location == 1.0) {
            REQUIRE(parts.size() == 1);
            assert_same(curve, parts[0], 10);
        } else {
            assert_same(curve, parts[0], 10, 0.0, split_location, 0.0, split_location);
            assert_same(curve, parts[1], 10, split_location, 1.0, split_location, 1.0);
        }
    }

    SECTION("BSpline debug" ){
        Eigen::Matrix<Scalar, 7, 3> control_pts;
        control_pts << 
       31.75,            0,  12.7, 
       31.75,     -54.9926,  12.7,
     -15.875,     -27.4963,  12.7,
       -63.5, -7.77651e-15,  12.7,
     -15.875,      27.4963,  12.7,
       31.75,      54.9926,  12.7,
       31.75,  7.77651e-15,  12.7;
        Eigen::Matrix<Scalar, 10, 1> knots;
        knots << 0, 0, 0,
              2.0944, 2.0944, 4.18879, 4.18879, 
              6.28319, 6.28319, 6.28319;

        BSpline<Scalar, 3, 2, true> curve;
        curve.set_control_points(control_pts);
        curve.set_knots(knots);

        Scalar split_location = 0.0;
        SECTION("Beginning") {
            split_location = 0.1;
        }
        SECTION("Middle") {
            split_location = 2.5;
        }
        SECTION("2/3") {
            split_location = 3. + 2.0/3.0;
        }
        SECTION("End") {
            split_location = 6.0;
        }
        Scalar t_min = curve.get_domain_lower_bound();
        Scalar t_max = curve.get_domain_upper_bound();
        const auto parts = split(curve, split_location);
            assert_same(curve, parts[0], 10, t_min, split_location, t_min, split_location);
            assert_same(curve, parts[1], 10, split_location, t_max, split_location, t_max);
    }



    SECTION("NURBS degree 2") {
        Eigen::Matrix<Scalar, 8, 2> ctrl_pts;
        ctrl_pts << 0.0, 0.0,
                 0.0, 1.0,
                 1.0, 1.0,
                 1.0, 0.0,
                 2.0, 0.0,
                 2.0, 1.0,
                 3.0, 1.0,
                 3.0, 0.0;
        Eigen::Matrix<Scalar, 11, 1> knots;
        knots << 0.0, 0.0, 0.0,
              1.0, 2.0, 3.0, 4.0, 5.0,
              6.0, 6.0, 6.0;
        Eigen::Matrix<Scalar ,8, 1> weights;
        weights << 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5;

        NURBS<Scalar, 2, 2, true> curve;
        curve.set_control_points(ctrl_pts);
        curve.set_knots(knots);
        curve.set_weights(weights);
        curve.initialize();
        // save_svg("test.svg", curve);

        Scalar split_location = 0.0;
        SECTION("Beginning") {
            split_location = 0.0;
        }
        SECTION("End") {
            split_location = 6.0;
        }
        SECTION("Half") {
            split_location = 3.0;
        }
        SECTION("Between 2 knots") {
            split_location = 0.5;
        }

        const auto parts = split(curve, split_location);
        if (split_location == 0.0) {
            REQUIRE(parts.size() == 1);
            assert_same(curve, parts[0], 10);
        } else if (split_location == 6.0) {
            REQUIRE(parts.size() == 1);
            assert_same(curve, parts[0], 10);
        } else {
            assert_same(curve, parts[0], 10, 0.0, split_location, 0.0, split_location);
            assert_same(curve, parts[1], 10, split_location, 6.0, split_location, 6.0);
        }
    }

    SECTION("BSpline loop") {
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

        const auto t_min = curve.get_domain_lower_bound();
        const auto t_max = curve.get_domain_upper_bound();
        Scalar split_location = 0.0;

        SECTION("Beginning") {
            split_location = t_min;
            const auto parts = split(curve, split_location);
            REQUIRE(parts.size() == 1);
            assert_same(curve, parts[0], 10);
        }

        SECTION("End") {
            split_location = t_max;
            const auto parts = split(curve, split_location);
            REQUIRE(parts.size() == 1);
            assert_same(curve, parts[0], 10);
        }

        SECTION("Half") {
            split_location = 0.5;
            const auto parts = split(curve, split_location);
            REQUIRE(parts.size() == 2);

            assert_same(curve, parts[0], 10, t_min, split_location, t_min, split_location);
            assert_same(curve, parts[1], 10, split_location, t_max, split_location, t_max);
        }

        SECTION("One knot before the end") {
            split_location = 13.0/17;
            const auto parts = split(curve, split_location);
            REQUIRE(parts.size() == 2);

            assert_same(curve, parts[0], 10, t_min, split_location, t_min, split_location);
            assert_same(curve, parts[1], 10, split_location, t_max, split_location, t_max);
        }
    }
}
TEST_CASE("Test patch splitting", "[split][patch]"){
  using Scalar = double;
  using namespace nanospline;
    SECTION("Bezier Patch degree 3") {
        const int degree_u = 3;
        const int degree_v = 3;
        const int dim = 3;
        const int num_control_pts = (degree_u+1)*(degree_v+1);

        BezierPatch<Scalar, dim, degree_u, degree_v> patch;
        Eigen::Matrix<Scalar, num_control_pts, dim> control_grid;
        for (int i = 0; i <= degree_u; i++) {
            for (int j = 0; j <= degree_v; j++) {
                control_grid.row(i * (degree_v+1) + j) << j, i, ((i + j) % 2 == 0) ? -1 : 1;
            }
        }
        patch.set_control_grid(control_grid);
        patch.initialize();
        validate_bezier_patch_splitting(patch);
    }
    SECTION("Bezier Patch random patches degree 5") {
        const int degree_u = 4;
        const int degree_v = 4;
        const int dim = 3;
        const int num_control_pts = (degree_u+1)*(degree_v+1);
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<Scalar> dist(0, 1);
        
        BezierPatch<Scalar, dim, degree_u, degree_v> patch;
        Eigen::Matrix<Scalar, num_control_pts, dim> control_grid;
        for (int i = 0; i <= degree_u; i++) {
            for (int j = 0; j <= degree_v; j++) {
                control_grid.row(i * (degree_v+1) + j) << dist(generator), dist(generator),dist(generator);
            }
        }
       
        patch.set_control_grid(control_grid);
        patch.initialize();
        validate_bezier_patch_splitting(patch);
    }
    SECTION("Bezier Patch mixed degree") {
        const int degree_u = 3;
        const int degree_v = 5;
        const int dim = 3;
        const int num_control_pts = (degree_u+1)*(degree_v+1);

        BezierPatch<Scalar, dim, degree_u, degree_v> patch;
        Eigen::Matrix<Scalar, num_control_pts, dim> control_grid;
        for (int i = 0; i <= degree_u; i++) {
            for (int j = 0; j <= degree_v; j++) {
                control_grid.row(i * (degree_v+1) + j) << j, i, ((i + j) % 2 == 0) ? -1 : 1;
            }
        }
        patch.set_control_grid(control_grid);
        patch.initialize();
        validate_bezier_patch_splitting(patch);
    }
    SECTION("Bilinear BSpline patch non-planar") {
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

        validate_bspline_patch_splitting(patch);
        
    
    }
    SECTION("Cubic BSpline patch"){
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

        validate_bspline_patch_splitting(patch);
    }
    
    SECTION("Degree 1 BSpline patch") {
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
        
        validate_bspline_patch_splitting(patch);
    }
    SECTION("Mixed degree BSpline"){
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

        validate_bspline_patch_splitting(patch);
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
        validate_bspline_patch_splitting(patch);
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
        validate_bspline_patch_splitting(patch);

    }
    SECTION("RationalBezier cubic patch") {
      RationalBezierPatch<Scalar, 3, 3, 3> patch;
      Eigen::Matrix<Scalar, 16, 3> control_grid;
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          control_grid.row(i * 4 + j) << j, i, ((i + j) % 2 == 0) ? -1 : 1;
        }
      }
      patch.set_control_grid(control_grid);

      Eigen::Matrix<Scalar, 16, 1> weights;
      SECTION("Uniform weight") {
        weights.setConstant(1);
        patch.set_weights(weights);
        patch.initialize();
        validate_bezier_patch_splitting(patch);
      }
      SECTION("Non-uniform weight") {
        weights.setConstant(1);
        weights[5] = 2.0;
        weights[6] = 2.0;
        weights[9] = 2.0;
        weights[10] = 2.0;
        patch.set_weights(weights);
        patch.initialize();
        validate_bezier_patch_splitting(patch);
      }
    }

    SECTION("Bilinear patch non-planar") {
        NURBSPatch<Scalar, 3, 1, 1> patch;
        Eigen::Matrix<Scalar, 4, 3> control_grid;
        control_grid <<
            0.0, 0.0, 0.0,
            0.0, 1.0, 1.0,
            1.0, 0.0, 1.0,
            1.0, 1.0, 0.0;
        patch.set_control_grid(control_grid);

        Eigen::Matrix<Scalar, 4, 1> weights;
        weights.setConstant(2.0);
        patch.set_weights(weights);

        Eigen::Matrix<Scalar, 4, 1> knots_u, knots_v;
        knots_u << 0.0, 0.0, 1.0, 1.0;
        knots_v << 0.0, 0.0, 1.0, 1.0;
        patch.set_knots_u(knots_u);
        patch.set_knots_v(knots_v);

        patch.initialize();
            validate_bspline_patch_splitting(patch);

    }

    SECTION("Cubic patch") {
        NURBSPatch<Scalar, 3, 3, 3> patch;
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

        Eigen::Matrix<Scalar, 16, 1> weights;
        weights.setOnes();
        SECTION("Uniform weight") {
            weights.setConstant(2.0);
            
            patch.set_weights(weights);
            patch.initialize();
            validate_bspline_patch_splitting(patch);
        }
        SECTION("Non-uniform weight") {
            weights[5] = 2.0;
            weights[6] = 2.0;
            weights[9] = 2.0;
            weights[10] = 2.0;
            
            patch.set_weights(weights);
            patch.initialize();
            validate_bspline_patch_splitting(patch);
        }

    }

    SECTION("Mixed degree") {
        NURBSPatch<Scalar, 3, -1, -1> patch;
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
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> weights(14, 1);
        weights << 1.0, 1.0,
                   0.5, 0.5,
                   1.0, 1.0,
                   0.5, 0.5,
                   1.0, 1.0,
                   0.5, 0.5,
                   1.0, 1.0;
        patch.set_weights(weights);
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
        validate_bspline_patch_splitting(patch);

    }
}


