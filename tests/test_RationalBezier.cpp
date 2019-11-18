#include <catch2/catch.hpp>

#include <nanospline/RationalBezier.h>

TEST_CASE("RationalBezier", "[rational][bezier]") {
    using namespace nanospline;

    SECTION("Generic degree 0") {
        Eigen::Matrix<float, 1, 2> control_pts;
        control_pts << 0.0, 0.1;
        Eigen::Matrix<float, 1, 1> weights;
        weights << 1.0;

        RationalBezier<float, 2, 0, true> curve;
        curve.set_control_points(control_pts);
        curve.set_weights(weights);
        curve.initialize();

        auto start = curve.evaluate(0);
        auto mid = curve.evaluate(0.5);
        auto end = curve.evaluate(1);

        REQUIRE((start-control_pts.row(0)).norm() == Approx(0.0));
        REQUIRE((end-control_pts.row(0)).norm() == Approx(0.0));
        REQUIRE((mid-control_pts.row(0)).norm() == Approx(0.0));
    }

}
