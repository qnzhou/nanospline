#include <catch2/catch.hpp>
#include <iostream>

#include <nanospline/Bezier.h>
#include <nanospline/Circle.h>
#include <nanospline/Ellipse.h>
#include <nanospline/NURBS.h>
#include <nanospline/RevolutionPatch.h>

#include "validation_utils.h"

TEST_CASE("RevolutionPatch", "[revolution_patch][primitive]")
{
    using namespace nanospline;
    using Scalar = double;

    SECTION("Simple")
    {
        Eigen::Matrix<Scalar, 4, 3> control_pts;
        control_pts << 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 0.0;
        Bezier<Scalar, 3, 3> profile;
        profile.set_control_points(control_pts);

        RevolutionPatch<Scalar, 3> patch;
        patch.set_profile(&profile);
        patch.set_axis({0, 1, 0});
        patch.initialize();
        REQUIRE(patch.get_dim() == 3);

        validate_derivative(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10);
        validate_inverse_evaluation_3d(patch, 10, 10, {0, 2 * M_PI, 0, 1});
    }

    SECTION("Torus")
    {
        Circle<Scalar, 3> profile;
        Eigen::Matrix<Scalar, 2, 3> frame;
        frame << 0, 0, 1, 0, 1, 0;
        profile.set_frame(frame);
        profile.set_radius(0.2);
        profile.set_center({0, 1, 0});
        profile.initialize();

        RevolutionPatch<Scalar, 3> patch;
        patch.set_profile(&profile);
        patch.initialize();
        REQUIRE(patch.get_dim() == 3);

        validate_derivative(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10);
        validate_inverse_evaluation_3d(patch, 10, 10, {0, 2 * M_PI, 0, 1});
    }

    SECTION("Orthogonal revolution")
    {
        Ellipse<Scalar, 3> profile;
        profile.set_major_radius(2);
        profile.set_minor_radius(1);
        profile.set_center({-5, -5, 0});
        Eigen::Matrix<Scalar, 2, 3> frame;
        frame << 1, 0, 0, 0, 1, 0;
        profile.set_frame(frame);
        profile.initialize();
        REQUIRE(profile.get_periodic());

        RevolutionPatch<Scalar, 3> patch;
        patch.set_profile(&profile);

        Eigen::Matrix<Scalar, 1, 3> axis(0, -1, 0);
        patch.set_axis(axis.normalized());
        patch.set_v_lower_bound(profile.get_domain_lower_bound());
        patch.set_v_upper_bound(profile.get_domain_upper_bound());
        patch.initialize();

        REQUIRE(patch.get_dim() == 3);
        REQUIRE(patch.get_periodic_u());
        REQUIRE(patch.get_periodic_v());

        validate_derivative(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10);
        validate_inverse_evaluation_3d(patch, 10, 10, {0, 2 * M_PI, 0, 1});
    }

    SECTION("Non-orthogonal revolution")
    {
        Ellipse<Scalar, 3> profile;
        profile.set_major_radius(2);
        profile.set_minor_radius(1);
        profile.set_center({-5, -5, 0});
        Eigen::Matrix<Scalar, 2, 3> frame;
        frame << 1, 0, 0, 0, 1, 0;
        profile.set_frame(frame);
        profile.initialize();
        REQUIRE(profile.get_periodic());

        RevolutionPatch<Scalar, 3> patch;
        patch.set_profile(&profile);

        Eigen::Matrix<Scalar, 1, 3> axis(0, -1, -1);
        patch.set_axis(axis.normalized());
        patch.set_v_lower_bound(profile.get_domain_lower_bound());
        patch.set_v_upper_bound(profile.get_domain_upper_bound());
        patch.initialize();

        REQUIRE(patch.get_dim() == 3);
        REQUIRE(patch.get_periodic_u());
        REQUIRE(patch.get_periodic_v());

        validate_derivative(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10);
        validate_inverse_evaluation_3d(patch, 10, 10, {0, 2 * M_PI, 0, 1});
    }

    SECTION("Debug")
    {
        NURBS<Scalar, 3> profile;
        Eigen::Matrix<Scalar, 4, 3, Eigen::RowMajor> control_pts;
        control_pts << 16.6568565701552, 41.274999999997, 8.36688850362814, 16.1918885852888,
            41.274999999997, 8.36688850362809, 15.8631065701552, 41.274999999997, 8.695670518761691,
            15.863106570155098, 41.274999999997, 9.16063850362803;
        Eigen::Matrix<Scalar, 8, 1> knots;
        knots << 0, 0, 0, 0, 1, 1, 1, 1;
        Eigen::Matrix<Scalar, 4, 1> weights;
        weights << 1.0, 0.804737854124372, 0.804737854124372, 1.0;
        profile.set_control_points(control_pts);
        profile.set_knots(knots);
        profile.set_weights(weights);
        profile.initialize();

        RevolutionPatch<Scalar, 3> patch;
        patch.set_profile(&profile);
        Eigen::Matrix<Scalar, 1, 3> location(
            16.656856570156297, 41.2749999999981, 8.99974539336768e-12);
        patch.set_location(location);
        Eigen::Matrix<Scalar, 1, 3> axis(1.0, -1.33152019593955e-13, 1.33152019593937e-13);
        patch.set_axis(axis);
        patch.set_u_lower_bound(0);
        patch.set_u_upper_bound(2 * M_PI);
        patch.set_v_lower_bound(9.0e-15);
        patch.set_v_upper_bound(1.0);
        patch.initialize();

        REQUIRE(patch.get_periodic_u());
        REQUIRE(!patch.get_periodic_v());

        validate_derivative(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10);
        validate_inverse_evaluation_3d(patch, 10, 10, {0, 2 * M_PI, 0, 1});

        SECTION("Query 1")
        {
            Eigen::Matrix<Scalar, 1, 3> q(15.863106570154715, 38.035987883037649, 8.5689057698641271);
            auto r = patch.inverse_evaluate(q, 3.1415926535897931, 6.2831853071795862, 0.1894796106285625, 1);
            const auto& uv = std::get<0>(r);
            auto p = patch.evaluate(uv[0], uv[1]);

            REQUIRE((p-q).norm() > 3);
        }

        SECTION("Query 2")
        {
            Eigen::Matrix<Scalar, 1, 3> q(15.863106570154715, 38.035987883037649, 8.5689057698641271);
            auto r = patch.inverse_evaluate(q, 0, 3.1415926535897931, 0.1894796106285625, 1);
            const auto& uv = std::get<0>(r);
            auto p = patch.evaluate(uv[0], uv[1]);

            REQUIRE((p-q).norm() == Approx(0).margin(1e-5));
        }
    }
}
