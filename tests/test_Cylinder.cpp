#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

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

    SECTION("Debug")
    {
        Cylinder<Scalar, 3> patch;
        patch.set_radius(2);
        patch.set_location({2.44929e-16, 17, 31.8355});
        Eigen::Matrix<Scalar, 3, 3> frame;
        frame << 0, 0, 1, -1, 0, 0, 0, -1, 0;
        patch.set_frame(frame);
        patch.set_u_lower_bound(0);
        patch.set_u_upper_bound(6.283185307179588);
        patch.set_v_lower_bound(-7.011050944326399);
        patch.set_v_upper_bound(5.167893780685986);
        patch.initialize();

        REQUIRE(patch.get_dim() == 3);
        REQUIRE(patch.get_periodic_u());

        validate_derivative(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10);
        validate_inverse_evaluation_3d(patch, 10, 10);

        SECTION("Query 1")
        {
            Eigen::Matrix<Scalar, 1, 3> q(
                4.8985871965894059e-16, 24.011050944326399, 29.835497046498102);
            auto uv = std::get<0>(patch.inverse_evaluate(q,
                -6.1575216010359952,
                0.12566370614359279,
                -7.0110509443263993,
                -0.92157858182020647));
            auto p = patch.evaluate(uv[0], uv[1]);
            REQUIRE_THAT((p - q).norm(), Catch::Matchers::WithinAbs(0, 1e-3));
        }

        SECTION("Query 2")
        {
            Eigen::Matrix<Scalar, 1, 3> q(
                -0.25066646712860841, 24.011050944326399, 29.851267643869146);
            auto uv = std::get<0>(patch.inverse_evaluate(q,
                -6.1575216010359952,
                0.12566370614359279,
                -7.0110509443263993,
                -0.92157858182020647));
            auto p = patch.evaluate(uv[0], uv[1]);

            REQUIRE_THAT((p - q).norm(), Catch::Matchers::WithinAbs(0, 1e-3));
        }
    }
}
