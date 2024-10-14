#include <nanospline/Torus.h>

#include "validation_utils.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEST_CASE("Torus", "[Torus][primitive]")
{
    using namespace nanospline;
    using Scalar = double;

    SECTION("Simple 3D")
    {
        Torus<Scalar, 3> patch;
        patch.initialize();
        REQUIRE(patch.get_dim() == 3);

        validate_derivative(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10);
        validate_inverse_evaluation_3d(patch, 10, 10);
    }

    SECTION("Paritial Torus")
    {
        Torus<Scalar, 3> patch;
        patch.set_major_radius(10.0);
        patch.set_minor_radius(1.0);
        patch.set_u_lower_bound(2 * M_PI - 1);
        patch.set_u_upper_bound(2 * M_PI + 1);
        patch.set_v_lower_bound(M_PI - 1);
        patch.set_v_upper_bound(M_PI + 1);
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
        Torus<Scalar, 3> patch;
        patch.set_major_radius(16);
        patch.set_minor_radius(5);
        patch.set_location({117.211201272414, 11.0, 0.874477237813336});
        patch.set_u_lower_bound(0);
        patch.set_u_upper_bound(2 * M_PI);
        patch.set_v_lower_bound(M_PI / 2);
        patch.set_v_upper_bound(M_PI * 1.5);
        Eigen::Matrix<Scalar, 3, 3> frame;
        frame << 0, 0, 1, 1, 0, 0, 0, 1, 0;
        patch.set_frame(frame);
        patch.initialize();
        REQUIRE(patch.get_dim() == 3);

        validate_derivative(patch, 10, 10);
        validate_inverse_evaluation(patch, 10, 10);
        validate_inverse_evaluation_3d(patch, 10, 10);

        SECTION("Query 1")
        {
            Eigen::Matrix<Scalar, 1, 3> q({117.211201272414, 6, -15.125522762186664});
            auto r = patch.inverse_evaluate(q,
                -6.2203534541077907,
                0.062831853071795951,
                1.5707963267948966,
                7.8539816339744828);
            const auto& uv = std::get<0>(r);
            auto p = patch.evaluate(uv[0], uv[1]);

            REQUIRE(std::get<1>(r));
            REQUIRE_THAT((q - p).norm(), Catch::Matchers::WithinAbs(0, 1e-6));
        }

        SECTION("Query 2")
        {
            Eigen::Matrix<Scalar, 1, 3> q({118.21584958488302, 6, -15.093950417039009});
            auto r = patch.inverse_evaluate(q,
                -9.3619461076975838,
                -3.0787608005179976,
                1.5707963267948966,
                7.8539816339744828);
            const auto& uv = std::get<0>(r);
            auto p = patch.evaluate(uv[0], uv[1]);

            REQUIRE(std::get<1>(r));
            REQUIRE_THAT((q - p).norm(), Catch::Matchers::WithinAbs(0, 1e-6));
        }
    }
}
