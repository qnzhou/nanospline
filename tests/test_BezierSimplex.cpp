#include <catch2/catch.hpp>

#include <nanospline/BezierSimplex.h>
#include <iostream>

TEST_CASE("BezierSimplex", "[nonrational][bezier_simplex]") {
    using namespace nanospline;
    using Scalar = double;

    BezierSimplex<Scalar, 2> bezier_triangle;
    REQUIRE(bezier_triangle.get_degree() == 3);
    REQUIRE(bezier_triangle.get_simplex_dim() == 2);
    REQUIRE(bezier_triangle.get_ordinate_dim() == -1);
    REQUIRE(bezier_triangle.get_num_control_points() == 10);

    auto ctrl_pts = bezier_triangle.get_control_points();
    std::cout << ctrl_pts << std::endl;
}
