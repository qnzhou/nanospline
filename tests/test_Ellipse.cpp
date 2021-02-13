#include <catch2/catch.hpp>

#include <nanospline/Ellipse.h>

#include "validation_utils.h"

TEST_CASE("Ellipse", "[primitive][ellipse]")
{
    using namespace nanospline;
    using Scalar = double;

    auto check_Ellipse = [](const auto& curve) {
        REQUIRE(curve.get_degree() == -1);
        REQUIRE(curve.is_closed(0, 1e-6));
        REQUIRE(curve.get_periodic());

        validate_derivatives(curve, 10);
        validate_2nd_derivatives(curve, 10);
        validate_approximate_inverse_evaluation(curve, 10);
    };

    SECTION("2D Ellipse")
    {
        Ellipse<Scalar, 2> curve;
        curve.set_major_radius(1);
        curve.set_minor_radius(0.5);
        check_Ellipse(curve);
    }

    SECTION("3D Ellipse")
    {
        Ellipse<Scalar, 3> curve;
        curve.set_major_radius(2);
        curve.set_minor_radius(0.1);
        curve.set_center({1, 1, 1});
        check_Ellipse(curve);
    }
}
