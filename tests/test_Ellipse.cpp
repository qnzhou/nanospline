#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <nanospline/Ellipse.h>

#include "validation_utils.h"

TEST_CASE("Ellipse", "[primitive][ellipse]")
{
    using namespace nanospline;
    using Scalar = double;

    auto check_inverse_evaluation_orthogonality = [](const auto& curve, int N) {
        Scalar R = curve.get_major_radius();
        for (int i = 0; i <= N; i++) {
            Scalar theta = (Scalar)i / (Scalar)N *
                               (curve.get_domain_upper_bound() - curve.get_domain_lower_bound()) +
                           curve.get_domain_lower_bound();
            const auto& frame = curve.get_frame();
            auto p = curve.get_center();
            p += std::cos(theta) * 1.1 * R * frame.row(0);
            p += std::sin(theta) * 1.1 * R * frame.row(1);

            auto t = curve.inverse_evaluate(p);
            auto v1 = (curve.evaluate(t) - p).normalized();
            auto v2 = curve.evaluate_derivative(t).normalized();

            if (t > curve.get_domain_lower_bound() && t < curve.get_domain_upper_bound()) {
                REQUIRE_THAT(v1.dot(v2), Catch::Matchers::WithinAbs(0, 1e-6));
            }
        }
    };

    auto check_Ellipse = [&](const auto& curve) {
        REQUIRE(curve.get_degree() == -1);

        validate_derivatives(curve, 10);
        validate_2nd_derivatives(curve, 10);
        validate_approximate_inverse_evaluation(curve, 10);
        check_inverse_evaluation_orthogonality(curve, 32);
    };

    SECTION("2D Ellipse")
    {
        Ellipse<Scalar, 2> curve;
        curve.set_major_radius(1);
        curve.set_minor_radius(0.5);
        curve.initialize();
        REQUIRE(curve.is_closed(0, 1e-6));
        REQUIRE(curve.get_periodic());
        check_Ellipse(curve);
    }

    SECTION("2D Ellipse Arc")
    {
        Ellipse<Scalar, 2> curve;
        curve.set_major_radius(5);
        curve.set_minor_radius(0.25);
        curve.set_center({-2, 1});
        curve.set_domain_lower_bound(2 * M_PI - 1);
        curve.set_domain_upper_bound(2 * M_PI + 1);
        curve.initialize();
        check_Ellipse(curve);
    }

    SECTION("3D Ellipse")
    {
        Ellipse<Scalar, 3> curve;
        curve.set_major_radius(2);
        curve.set_minor_radius(0.1);
        curve.set_center({1, 1, 1});
        curve.initialize();
        REQUIRE(curve.is_closed(0, 1e-6));
        REQUIRE(curve.get_periodic());
        check_Ellipse(curve);
    }

    SECTION("3D Ellipse Arc")
    {
        Ellipse<Scalar, 3> curve;
        curve.set_major_radius(100);
        curve.set_minor_radius(0.01);
        curve.set_center({1e-12, -100, 300});
        curve.set_domain_lower_bound(M_PI);
        curve.set_domain_upper_bound(2*M_PI);
        curve.initialize();
        check_Ellipse(curve);
    }
}
