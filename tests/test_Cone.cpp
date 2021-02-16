#include <catch2/catch.hpp>
#include <iostream>

#include <nanospline/Cone.h>

#include "validation_utils.h"

TEST_CASE("Cone", "[Cone][primitive]")
{
    using namespace nanospline;
    using Scalar = double;

    SECTION("Simple 3D")
    {
        Cone<Scalar, 3> patch;
        patch.initialize();
        REQUIRE(patch.get_dim() == 3);

        validate_derivative(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10);
        validate_inverse_evaluation_3d(patch, 10, 10);
    }

    SECTION("Paritial Cone")
    {
        Cone<Scalar, 3> patch;
        patch.set_angle(0.2);
        patch.set_radius(0.0);
        patch.set_u_lower_bound(2 * M_PI - 1);
        patch.set_u_upper_bound(2 * M_PI + 1);
        patch.set_v_lower_bound(0);
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

        {
            // Test points on the other side of the cone.
            auto p = patch.evaluate(M_PI, -1);
            auto uv = patch.inverse_evaluate(p,
                patch.get_u_lower_bound(),
                patch.get_u_upper_bound(),
                patch.get_v_lower_bound(),
                patch.get_v_upper_bound());
            auto q = patch.evaluate(uv[0], uv[1]);
            REQUIRE(q == patch.get_location());
        }
    }
}
