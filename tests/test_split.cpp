#include <catch2/catch.hpp>

#include <nanospline/split.h>
#include <nanospline/BezierPatch.h>
#include <nanospline/BSplinePatch.h>
#include <nanospline/save_svg.h>
#include <nanospline/forward_declaration.h>
#include "validation_utils.h"
#include <iostream>
template<typename PatchType>
void validate_patch_splitting(PatchType patch){
  using Scalar = typename PatchType::Scalar;
    SECTION("Degenerate split: u,v=(0,0)") {
        Scalar u = 0.;
        Scalar v = 0.;
        const auto subpatches = patch.split(u, v);
        cout << "num split " << subpatches.size() << endl;
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
        /*if (split_location == 0.5) {
        cout << "parent" << endl;
        cout << curve.get_control_points() << endl;

        cout << "child <.5" << endl;
        cout << parts[0].get_control_points() << endl;
        cout << "child >.5 " << endl;
        cout << parts[1].get_control_points() << endl;
        exit(0);
        }*/
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
        using std::cout;
        using std::endl;
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
        validate_patch_splitting(patch);
    }
    SECTION("Bezier Patch random patches degree 5") {
        const int degree_u = 3;
        const int degree_v = 3;
        const int dim = 3;
        const int num_control_pts = (degree_u+1)*(degree_v+1);
        // TODO bug in Catch preventing a loop to try multiple random patches.
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
        //validate_patch_splitting(patch);
    }
    SECTION("Bezier Patch mixed degree") {
        using std::cout;
        using std::endl;
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
        //validate_patch_splitting(patch);
    }
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

        //validate_patch_splitting(patch);
    SECTION("Split: u=.5") {
        Scalar u = .5;
        const auto subpatches = patch.split_u(u);
        assert_same(patch, subpatches[0], 10, 
                0., u, 0., 1.,
                0., u, 0., 1.);
        assert_same(patch, subpatches[1], 10, 
                u, 1., 0., 1.,
                u, 1., 0., 1.);
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
                    0., u, 0., v);
            assert_same(patch, subpatches[2], 10, 
                    u,  1., 0., v,
                    u, 1., 0., v);
            assert_same(patch, subpatches[1], 10, 
                    0., u,   v, 1.,
                    0., u, v, 1.);

            assert_same(patch, subpatches[3], 10, 
                    u,  1.,  v, 1.,
                    u, 1., v, 1.);
        }
    }
    SECTION("Split: v=.5") {
        Scalar v = .5;
        const auto subpatches = patch.split_v(v);
        assert_same(patch, subpatches[0], 10, 
                0., 1., 0., v,
                0., 1., 0., v);
        assert_same(patch, subpatches[1], 10, 
                0., 1., v,  1.,
                0., 1., v, 1.);
    }
    
    }
}


