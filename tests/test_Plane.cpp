#include <catch2/catch.hpp>
#include <iostream>

#include <nanospline/Plane.h>

#include "validation_utils.h"

TEST_CASE("Plane", "[plane][primitive]")
{
    using namespace nanospline;
    using Scalar = double;

    SECTION("Simple 3D")
    {
        Plane<Scalar, 3> patch;
        patch.initialize();
        REQUIRE(patch.get_dim() == 3);

        validate_derivative(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10);
        validate_inverse_evaluation_3d(patch, 10, 10);
    }

    SECTION("Simple 2D")
    {
        Plane<Scalar, 2> patch;
        patch.set_u_lower_bound(-1);
        patch.set_u_upper_bound(1);
        patch.set_location({-1, -1});
        Eigen::Matrix<Scalar, 2, 2> frame;
        frame << 0, 1, -2, 0;
        patch.set_frame(frame);
        patch.initialize();
        REQUIRE(patch.get_dim() == 2);

        validate_derivative(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10);
    }
}
