#include <catch2/catch.hpp>
#include <iostream>

#include <nanospline/BezierPatch.h>

TEST_CASE("BeizerPatch", "[nonrational][bezier_patch]") {
    using namespace nanospline;
    using Scalar = double;

    SECTION("Bilinear patch") {
        BezierPatch<Scalar, 3, 1, 1> patch;
        Eigen::Matrix<Scalar, 4, 3> control_grid;
        control_grid <<
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            1.0, 1.0, 0.0;
        patch.set_control_grid(control_grid);

        REQUIRE(patch.get_degree_u() == 1);
        REQUIRE(patch.get_degree_v() == 1);

        auto corner_00 = patch.evaluate(0.0, 0.0);
        auto corner_01 = patch.evaluate(0.0, 1.0);
        auto corner_11 = patch.evaluate(1.0, 1.0);
        auto corner_10 = patch.evaluate(1.0, 0.0);
        REQUIRE((corner_00 - control_grid.row(0)).norm() == Approx(0.0));
        REQUIRE((corner_10 - control_grid.row(1)).norm() == Approx(0.0));
        REQUIRE((corner_01 - control_grid.row(2)).norm() == Approx(0.0));
        REQUIRE((corner_11 - control_grid.row(3)).norm() == Approx(0.0));

        auto p_mid = patch.evaluate(0.5, 0.5);
        REQUIRE(p_mid[0] == Approx(0.5));
        REQUIRE(p_mid[1] == Approx(0.5));
        REQUIRE(p_mid[2] == Approx(0.0));
    }

    SECTION("Bilinear patch non-planar") {
        BezierPatch<Scalar, 3, 1, 1> patch;
        Eigen::Matrix<Scalar, 4, 3> control_grid;
        control_grid <<
            0.0, 0.0, 0.0,
            1.0, 0.0, 1.0,
            0.0, 1.0, 1.0,
            1.0, 1.0, 0.0;
        patch.set_control_grid(control_grid);

        REQUIRE(patch.get_degree_u() == 1);
        REQUIRE(patch.get_degree_v() == 1);

        auto corner_00 = patch.evaluate(0.0, 0.0);
        auto corner_01 = patch.evaluate(0.0, 1.0);
        auto corner_11 = patch.evaluate(1.0, 1.0);
        auto corner_10 = patch.evaluate(1.0, 0.0);
        REQUIRE((corner_00 - control_grid.row(0)).norm() == Approx(0.0));
        REQUIRE((corner_10 - control_grid.row(1)).norm() == Approx(0.0));
        REQUIRE((corner_01 - control_grid.row(2)).norm() == Approx(0.0));
        REQUIRE((corner_11 - control_grid.row(3)).norm() == Approx(0.0));

        auto p_mid = patch.evaluate(0.5, 0.5);
        REQUIRE(p_mid[0] == Approx(0.5));
        REQUIRE(p_mid[1] == Approx(0.5));
        REQUIRE(p_mid[2] == Approx(0.5));
    }
}
