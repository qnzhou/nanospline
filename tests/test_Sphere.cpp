#include <catch2/catch.hpp>
#include <iostream>

#include <nanospline/Sphere.h>

#include "validation_utils.h"

TEST_CASE("Sphere", "[sphere][primitive]")
{
    using namespace nanospline;
    using Scalar = double;

    SECTION("Simple 3D")
    {
        Sphere<Scalar, 3> patch;
        patch.initialize();
        REQUIRE(patch.get_dim() == 3);

        validate_derivative(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10);
        validate_inverse_evaluation_3d(patch, 10, 10);
    }

    SECTION("Paritial sphere")
    {
        Sphere<Scalar, 3> patch;
        patch.set_radius(0.5);
        patch.set_u_lower_bound(2 * M_PI - 1);
        patch.set_u_upper_bound(2 * M_PI + 1);
        patch.set_v_lower_bound(-1);
        patch.set_v_upper_bound(1);
        Eigen::Matrix<Scalar, 3, 3> frame;
        frame << 0, 1, 0, 0, 0, 1, 1, 0, 0;
        frame =
            Eigen::AngleAxis<Scalar>(1, Eigen::Matrix<Scalar, 3, 1>::Ones().normalized()) * frame;
        patch.set_frame(frame);
        patch.initialize();
        REQUIRE(patch.get_dim() == 3);

        validate_derivative(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10);
        validate_inverse_evaluation_3d(patch, 10, 10, {-M_PI + 0.1, M_PI - 0.1, -1.1, 1.1});
    }

    SECTION("Inverse evaluation with initial guess")
    {
        Sphere<Scalar, 3> patch;
        patch.initialize();
        REQUIRE(patch.get_dim() == 3);

        Eigen::Matrix<Scalar, 1, 3> q(2, 0, 0);
        Scalar initial_u = 0.5, initial_v = 0.5;

        auto r = patch.inverse_evaluate(q, initial_u, initial_v, -1, 1, -1, 1, 20, 1e-6);
        auto& uv = std::get<0>(r);
        bool converged = std::get<1>(r);
        REQUIRE(converged);
        REQUIRE(uv[0] == Approx(0).margin(1e-6));
        REQUIRE(uv[1] == Approx(0).margin(1e-6));
    }
}
