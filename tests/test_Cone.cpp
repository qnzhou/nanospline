#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
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

    SECTION("Simple 3D 2")
    {
        Cone<Scalar, 3> patch;
        patch.set_location({1, 2, 3});

        const Scalar c = std::sqrt(2) / 2;
        Eigen::Matrix<Scalar, 3, 3> frame;
        frame << 0, 0, 1, c, c, 0, -c, c, 0;
        patch.set_frame(frame);

        patch.set_radius(0.5);
        patch.set_angle(M_PI / 6);

        patch.initialize();

        REQUIRE(patch.get_dim() == 3);
        REQUIRE(patch.get_periodic_u());

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
            auto uv = std::get<0>(patch.inverse_evaluate(p,
                patch.get_u_lower_bound(),
                patch.get_u_upper_bound(),
                patch.get_v_lower_bound(),
                patch.get_v_upper_bound()));
            auto q = patch.evaluate(uv[0], uv[1]);
            REQUIRE(q == patch.get_location());
        }
    }

    SECTION("Singularity")
    {
        Cone<Scalar, 3> patch;
        Eigen::Matrix<Scalar, 1, 3> l(2.4492935982947e-16, 32.0, 31.8354970464981);
        patch.set_location(l);
        patch.set_radius(0);
        patch.set_angle(0.7853981634);
        patch.set_u_lower_bound(M_PI);
        patch.set_u_upper_bound(3 * M_PI);
        patch.set_v_lower_bound(0);
        patch.set_v_upper_bound(1.414213562373095);
        patch.initialize();

        validate_derivative(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10);
        validate_inverse_evaluation_3d(patch, 10, 10);

        // Ensure `l` is the singularity
        REQUIRE_THAT((patch.evaluate(0, 0) - l).norm(), Catch::Matchers::WithinAbs(0, 1e-6));
        REQUIRE_THAT(patch.evaluate_derivative_u(0, 0).norm(), Catch::Matchers::WithinAbs(0, 1e-6));
        REQUIRE_THAT((patch.evaluate(1, 0) -l).norm(), Catch::Matchers::WithinAbs(0, 1e-6));
        REQUIRE_THAT((patch.evaluate(M_PI, 0) - l).norm(), Catch::Matchers::WithinAbs(0, 1e-6));

        SECTION("Query at singularity") {
            Eigen::Matrix<Scalar, 1, 3> q1(l[0], l[1], l[2]);
            auto uv = std::get<0>(patch.inverse_evaluate(q1,
                patch.get_u_lower_bound(),
                patch.get_u_upper_bound(),
                patch.get_v_lower_bound(),
                patch.get_v_upper_bound()));
            REQUIRE_THAT(uv[1], Catch::Matchers::WithinAbs(0, 1e-6));
        }

        SECTION("Query outside of the cone") {
            Eigen::Matrix<Scalar, 1, 3> q1(l[0], l[1], l[2] - 1);
            auto uv = std::get<0>(patch.inverse_evaluate(q1,
                patch.get_u_lower_bound(),
                patch.get_u_upper_bound(),
                patch.get_v_lower_bound(),
                patch.get_v_upper_bound()));
            REQUIRE_THAT(uv[1], Catch::Matchers::WithinAbs(0, 1e-6));
        }

        SECTION("Query inside of the cone") {
            Eigen::Matrix<Scalar, 1, 3> q1(l[0], l[1], l[2] + 1);
            auto uv = std::get<0>(patch.inverse_evaluate(q1,
                patch.get_u_lower_bound(),
                patch.get_u_upper_bound(),
                patch.get_v_lower_bound(),
                patch.get_v_upper_bound()));
            REQUIRE(uv[1] > 0);
        }
    }
}
