#include <catch2/catch.hpp>
#include <iostream>

#include <nanospline/Bezier.h>
#include <nanospline/Circle.h>
#include <nanospline/ExtrusionPatch.h>

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
        frame << 1, 0, 0,
                 0, 1, 0;
        profile.set_frame(frame);
        profile.set_radius(0.2);
        profile.set_center({0, 0, 0});
        profile.initialize();

        ExtrusionPatch<Scalar, 3> patch;
        patch.set_profile(&profile);
        patch.set_u_upper_bound(M_PI);
        patch.set_v_upper_bound(10);
        patch.initialize();
        REQUIRE(patch.get_dim() == 3);

        validate_derivative(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10);
        validate_inverse_evaluation_3d(patch, 10, 10);
    }
}
