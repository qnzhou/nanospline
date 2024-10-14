#include <catch2/catch_test_macros.hpp>

#include <nanospline/Line.h>

#include "validation_utils.h"

TEST_CASE("Line", "[primitive][line]")
{
    using namespace nanospline;
    using Scalar = double;

    auto check_line = [](const auto& curve) {
        REQUIRE(curve.get_degree() == -1);
        REQUIRE(!curve.is_closed(0, 1e-6));
        REQUIRE(!curve.get_periodic());

        {
            validate_derivatives(curve, 10);
            validate_2nd_derivatives(curve, 10);
            validate_approximate_inverse_evaluation(curve, 10);
        }
    };

    SECTION("2D line")
    {
        Line<Scalar, 2> curve;
        curve.set_location({1, 1});
        curve.set_direction({1, 0});
        curve.set_domain_lower_bound(1);
        curve.set_domain_upper_bound(10);
        check_line(curve);
    }

    SECTION("3D line")
    {
        Line<Scalar, 3> curve;
        curve.set_location({0, 0, 0});
        curve.set_direction({1, 0, 1});
        curve.set_domain_lower_bound(1);
        curve.set_domain_upper_bound(2);
        check_line(curve);
    }
}
