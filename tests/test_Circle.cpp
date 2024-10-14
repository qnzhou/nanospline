#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <nanospline/Circle.h>

#include "validation_utils.h"

TEST_CASE("Circle", "[primitive][circle]")
{
    using namespace nanospline;
    using Scalar = double;

    auto check_circle = [](const auto& curve) {
        REQUIRE(curve.get_degree() == -1);
        REQUIRE(curve.is_closed(0, 1e-6));
        REQUIRE(curve.get_periodic());

        {
            auto p0 = curve.evaluate(0);
            auto p1 = curve.evaluate(M_PI);
            auto p2 = curve.evaluate(2*M_PI);
            auto c = (p0 + p1) / 2;
            REQUIRE_THAT((p0 - p2).norm(), Catch::Matchers::WithinAbs(0, 1e-6));
            auto p3 = curve.evaluate(M_PI/3);
            REQUIRE_THAT((p3 - c).norm(), Catch::Matchers::WithinAbs((p0 - c).norm(), 1e-6));
        }

        {
            validate_derivatives(curve, 10);
            validate_2nd_derivatives(curve, 10);
            validate_approximate_inverse_evaluation(curve, 10);
        }
    };

    SECTION("2D circle")
    {
        Circle<Scalar, 2> curve;
        curve.set_radius(1);
        curve.initialize();
        check_circle(curve);
    }

    SECTION("3D circle")
    {
        Circle<Scalar, 3> curve;
        curve.set_radius(2);
        curve.set_center({1, 1, 1});
        curve.initialize();
        check_circle(curve);
    }

    SECTION("Circle arc")
    {
        Circle<Scalar, 2> curve;
        curve.set_radius(1);
        curve.set_domain_lower_bound(0);
        curve.set_domain_upper_bound(M_PI);
        curve.initialize();

        REQUIRE(!curve.get_periodic());
        REQUIRE(!curve.is_closed());
        REQUIRE_THAT(curve.get_turning_angle(0, M_PI), Catch::Matchers::WithinAbs(M_PI, 1e-6));
    }

    SECTION("Degenerate circle")
    {
        Circle<Scalar, 3> curve;
        curve.set_radius(0);
        curve.set_center({-1, 1, 1});
        curve.initialize();
        check_circle(curve);
    }

}
