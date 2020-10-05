#include <catch2/catch.hpp>

#include <vector>

#include <nanospline/arc_length.h>
#include <nanospline/forward_declaration.h>
#include <nanospline/Bezier.h>
#include <nanospline/BSpline.h>
#include <nanospline/NURBS.h>
#include "validation_utils.h"

TEST_CASE("arc_length", "[arc_length]")
{
    using namespace nanospline;
    using Scalar = double;

    auto check_arc_length = [](auto& curve) {
        const auto t_min = curve.get_domain_lower_bound();
        const auto t_max = curve.get_domain_upper_bound();
        constexpr size_t N = 10;
        std::vector<Scalar> lengths(N + 1, 0.0);
        for (size_t i = 0; i <= N; i++) {
            Scalar t = (Scalar)i / (Scalar)N * (t_max - t_min) + t_min;
            lengths[i] = arc_length(curve, t);
        }

        for (size_t i = 0; i <= N; i++) {
            const Scalar t1 = (Scalar)i / (Scalar)N * (t_max - t_min) + t_min;
            const Scalar t2 = inverse_arc_length(curve, lengths[i]);
            REQUIRE(t1 == Approx(t2));
        }
    };

    SECTION("Bezier degree 3")
    {
        Eigen::Matrix<Scalar, 4, 2> ctrl_pts;
        ctrl_pts << 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0;
        Bezier<Scalar, 2, 3> curve;
        curve.set_control_points(ctrl_pts);

        check_arc_length(curve);
    }

    SECTION("BSpline degree 3")
    {
        Eigen::Matrix<Scalar, 10, 2> ctrl_pts;
        ctrl_pts << 1, 4, .5, 6, 5, 4, 3, 12, 11, 14, 8, 4, 12, 3, 11, 9, 15, 10, 17, 8;
        Eigen::Matrix<Scalar, 14, 1> knots;
        knots << 0, 0, 0, 0, 1.0 / 7, 2.0 / 7, 3.0 / 7, 4.0 / 7, 5.0 / 7, 6.0 / 7, 1, 1, 1, 1;

        BSpline<Scalar, 2, 3, true> curve;
        curve.set_control_points(ctrl_pts);
        curve.set_knots(knots);

        check_arc_length(curve);
    }

    SECTION("Quarter circle") {
        constexpr Scalar R = 12.1;
        Eigen::Matrix<Scalar, 1, 2> c(0.0, R);
        Eigen::Matrix<Scalar, 9, 2> control_pts;
        control_pts << 0.0, 0.0,
                    R, 0.0,
                    R, R,
                    R, 2*R,
                    0.0, 2*R,
                    -R, 2*R,
                    -R, R,
                    -R, 0.0,
                    0.0, 0.0;
        Eigen::Matrix<Scalar, 12, 1> knots;
        knots << 0.0, 0.0, 0.0,
              0.25, 0.25,
              0.5, 0.5,
              0.75, 0.75,
              1.0, 1.0, 1.0;
        Eigen::Matrix<Scalar, 9, 1> weights;
        weights << 1.0, sqrt(2)/2,
                1.0, sqrt(2)/2,
                1.0, sqrt(2)/2,
                1.0, sqrt(2)/2,
                1.0;

        NURBS<Scalar, 2, 2, true> curve;
        curve.set_control_points(control_pts);
        curve.set_knots(knots);
        curve.set_weights(weights);
        curve.initialize();

        check_arc_length(curve);
    }
}
