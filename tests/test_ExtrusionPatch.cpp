#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include <nanospline/Bezier.h>
#include <nanospline/Circle.h>
#include <nanospline/ExtrusionPatch.h>
#include <nanospline/NURBS.h>
#include <nanospline/save_msh.h>

#include "validation_utils.h"

TEST_CASE("ExtrusionPatch", "[extrusion_patch][primitive]")
{
    using namespace nanospline;
    using Scalar = double;

    SECTION("Simple")
    {
        Eigen::Matrix<Scalar, 4, 3> control_pts;
        control_pts << 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 0.0;
        Bezier<Scalar, 3, 3> profile;
        profile.set_control_points(control_pts);

        ExtrusionPatch<Scalar, 3> patch;
        patch.set_profile(&profile);
        patch.set_direction({0, 0, 1});
        patch.initialize();
        REQUIRE(patch.get_dim() == 3);

        validate_derivative(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10);
        validate_inverse_evaluation_3d(patch, 10, 10);
    }

    SECTION("Cylinder")
    {
        Circle<Scalar, 3> profile;
        Eigen::Matrix<Scalar, 2, 3> frame;
        frame << 1, 0, 0, 0, 1, 0;
        profile.set_frame(frame);
        profile.set_radius(0.2);
        profile.set_center({0, 0, 0});
        profile.initialize();

        ExtrusionPatch<Scalar, 3> patch;
        patch.set_profile(&profile);
        patch.set_u_upper_bound(M_PI);
        patch.set_v_upper_bound(10);
        patch.set_direction({0, 0, 1});
        patch.initialize();
        REQUIRE(patch.get_dim() == 3);

        validate_derivative(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10);
        validate_inverse_evaluation_3d(patch, 10, 10);
    }

    SECTION("Debug")
    {
        BSpline<Scalar, 3, 2> profile;
        Eigen::Matrix<Scalar, 3, 3, Eigen::RowMajor> control_pts;
        control_pts << 8.917513415217401, 33.0, 35.5111918712002, 8.917513415217401, 33.0,
            36.8296487375194, 8.917513415217401, 34.211731608475596, 37.3492976741733;
        profile.set_control_points(control_pts);
        Eigen::Matrix<Scalar, 1, 6> knots;
        knots << 0.0, 0.0, 0.0, 1.0, 1.0, 1.0;
        profile.set_knots(knots);
        profile.initialize();

        ExtrusionPatch<Scalar, 3> patch;
        patch.set_profile(&profile);
        patch.set_direction({-1, 0, 0});
        patch.set_u_lower_bound(0);
        patch.set_u_upper_bound(1);
        patch.set_v_lower_bound(-1.0e-15);
        patch.set_v_upper_bound(3.917513415217402);
        patch.initialize();

        validate_derivative(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10);
        validate_inverse_evaluation_3d(patch, 10, 10);
    }

    SECTION("Debug2")
    {
        NURBS<Scalar, 3, 3> profile;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 3, Eigen::RowMajor> control_pts(7, 3);
        control_pts << -33.7558788369837, -333.522706450385, 91.59932332812019, -33.720128739786496,
            -404.00740143555606, 91.8178753210735, 35.709419778735295, -404.00740143555606,
            78.8151510775912, 35.673669681537994, -333.522706450385, 78.5965990846379,
            35.6379195843407, -263.038011465212, 78.37804709168459, -33.7916289341811,
            -263.038011465212, 91.3807713351668, -33.7558788369837, -333.522706450385,
            91.59932332812019;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> knots(11, 1);
        knots << -0.5, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.5;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> weights(7, 1);
        weights << 1.0, 0.333333333333334, 0.333333333333334, 1.0, 0.333333333333334,
            0.333333333333334, 1.0;
        profile.set_control_points(control_pts);
        profile.set_knots(knots);
        profile.set_weights(weights);
        profile.initialize();
        validate_approximate_inverse_evaluation(profile, 10);

        ExtrusionPatch<Scalar, 3> patch;
        patch.set_profile(&profile);
        Eigen::Matrix<Scalar, 1, 3> direction(
            0.16143097665315304, 0.0027546189786395005, 0.9868801608356972);
        patch.set_direction(direction);
        patch.set_u_lower_bound(0.249935636023151);
        patch.set_u_upper_bound(0.501326429106788);
        patch.set_v_lower_bound(121.8883258185659);
        patch.set_v_upper_bound(161.70527576265616);
        patch.initialize();
        validate_derivative(patch, 10, 10);

        SECTION("Query 1")
        {
            Eigen::Matrix<Scalar, 1, 3> q(
                23.423596982756884, -368.31689893094199, 204.97723314984171);
            auto r = patch.inverse_evaluate(
                q, 0.249935636023151, 0.37563103256496955, 121.8883258185659, 141.79680079061171);
            const auto& uv = std::get<0>(r);
            auto p = patch.evaluate(uv[0], uv[1]);

            REQUIRE(std::get<1>(r));
            REQUIRE((p-q).norm() < 1e-6);
        }

        validate_inverse_evaluation(patch, 10, 10, 1e-6);
        validate_inverse_evaluation_3d(patch, 10, 10);
    }
}
