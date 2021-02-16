#include <catch2/catch.hpp>
#include <iostream>

#include <nanospline/Cylinder.h>

#include "validation_utils.h"

TEST_CASE("Cylinder", "[cylinder][primitive]")
{
    using namespace nanospline;
    using Scalar = double;

    SECTION("Simple 3D")
    {
        Cylinder<Scalar, 3> patch;
        patch.initialize();
        REQUIRE(patch.get_dim() == 3);

        validate_derivative(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10);
        validate_inverse_evaluation_3d(patch, 10, 10);
    }

    SECTION("Paritial cylinder")
    {
        Cylinder<Scalar, 3> patch;
        patch.set_radius(0.5);
        patch.set_u_lower_bound(2 * M_PI - 1);
        patch.set_u_upper_bound(2 * M_PI + 1);
        patch.set_v_lower_bound(-1);
        patch.set_v_upper_bound(100);
        Eigen::Matrix<Scalar, 3, 3> frame;
        frame << 0, 1, 0, 0, 0, 1, 1, 0, 0;
        frame =
            Eigen::AngleAxis<Scalar>(1, Eigen::Matrix<Scalar, 3, 1>::Ones().normalized()) * frame;
        patch.set_frame(frame);
        patch.initialize();
        REQUIRE(patch.get_dim() == 3);

        validate_derivative(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10);
        validate_inverse_evaluation_3d(patch, 10, 10);
    }
}