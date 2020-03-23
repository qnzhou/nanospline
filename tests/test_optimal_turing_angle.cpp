#include <catch2/catch.hpp>
#include <iostream>

#include <nanospline/optimal_turning_angle.h>
#include <nanospline/split.h>
#include <nanospline/save_svg.h>
#include <nanospline/forward_declaration.h>

TEST_CASE("opt_turn_angle", "[opt_turn_angle]") {
    using namespace nanospline;
    using Scalar = double;

    SECTION("Simple") {
        Eigen::Matrix<Scalar, 4, 2, Eigen::RowMajor> control_pts;
        control_pts << 0.0, 0.0,
                       0.0, 1.0,
                       1.0, 1.0,
                       1.0, 0.0;

        using Curve = Bezier<Scalar, 2, 3>;
        Curve curve;
        curve.set_control_points(control_pts);
        auto t = optimal_points_to_reduce_turning_angle(curve, false);

        REQUIRE(t.size() == 1);
        REQUIRE(t[0] == Approx(0.5));

        t = OptimalPointsToReduceTurningAngle<Curve>::compute(curve, false);
        REQUIRE(t.size() == 1);
    }

    SECTION("Simple")
    {
        Eigen::Matrix<Scalar, 4, 2, Eigen::RowMajor> control_pts;
        control_pts << 0.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
            1.0, 0.0;

        using Curve = Bezier<Scalar, 2, 3>;
        Curve curve;
        curve.set_control_points(control_pts);
        auto t = optimal_points_to_reduce_turning_angle(curve, false, 0.1, 0.4);

        REQUIRE(t.size() == 1);
        REQUIRE(t[0] == Approx(.2670068914));

        t = OptimalPointsToReduceTurningAngle<Curve>::compute(curve, false, 0.1, 0.4);
        REQUIRE(t.size() == 1);
    }
}
