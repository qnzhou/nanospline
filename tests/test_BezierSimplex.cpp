#include <catch2/catch.hpp>

#include <nanospline/BezierSimplex.h>
#include <iostream>

TEST_CASE("BezierSimplex", "[nonrational][bezier_simplex]")
{
    using namespace nanospline;
    using Scalar = double;

    auto finite_difference = [](auto&& simplex, auto&& b, auto&& dir) {
        constexpr Scalar eps = 1e-6;
        auto l = dir.norm();
        auto p1 = simplex.evaluate(b);
        auto p2 = simplex.evaluate(b + dir / l * eps);
        return ((p2 - p1) / eps * l).eval();
    };

    SECTION("Bezier triangle")
    {
        BezierSimplex<Scalar, 2, 2> bezier_triangle;
        REQUIRE(bezier_triangle.get_degree() == 2);
        REQUIRE(bezier_triangle.get_simplex_dim() == 2);
        REQUIRE(bezier_triangle.get_ordinate_dim() == -1);
        REQUIRE(bezier_triangle.get_num_control_points() == 6);

        SECTION("linear")
        {
            Eigen::Matrix<Scalar, 6, 2, Eigen::RowMajor> ordinates;
            ordinates << 0, 0, 0.5, 0, 0, 0.5, 1, 0, 0.5, 0.5, 0, 1;
            bezier_triangle.set_ordinates(ordinates);

            auto ctrl_pts = bezier_triangle.get_control_points();
            for (Eigen::Index i = 0; i < 6; i++) {
                auto p = bezier_triangle.evaluate(ctrl_pts.row(i));
                REQUIRE(p.isApprox(ordinates.row(i)));
            }

            Eigen::Matrix<Scalar, 1, 3> dir(-0.2, 0.3, -0.1);
            for (Eigen::Index i = 0; i < 6; i++) {
                Eigen::Matrix<Scalar, 1, 3> p = ctrl_pts.row(i);
                auto d_center = bezier_triangle.evaluate_directional_derivative(p, dir);
                auto d_center_diff = finite_difference(bezier_triangle, p, dir);
                REQUIRE((d_center - d_center_diff).norm() < 1e-4);
            }
        }

        SECTION("non-linear")
        {
            Eigen::Matrix<Scalar, 6, 1> ordinates;
            ordinates << 1, 0, 0, 1, 0, 1;
            bezier_triangle.set_ordinates(ordinates);

            auto ctrl_pts = bezier_triangle.get_control_points();
            for (Eigen::Index i = 0; i < 6; i++) {
                auto p = bezier_triangle.evaluate(ctrl_pts.row(i));
                if (i == 0 || i == 3 || i == 5) {
                    REQUIRE(p.isApprox(ordinates.row(i)));
                } else {
                    REQUIRE(!p.isApprox(ordinates.row(i)));
                }
            }

            Eigen::Matrix<Scalar, 1, 3> dir(1, 1, -2);
            for (Eigen::Index i = 0; i < 6; i++) {
                Eigen::Matrix<Scalar, 1, 3> p = ctrl_pts.row(i);
                auto d_center = bezier_triangle.evaluate_directional_derivative(p, dir);
                auto d_center_diff = finite_difference(bezier_triangle, p, dir);
                REQUIRE((d_center - d_center_diff).norm() < 1e-4);
            }
        }
    }

    SECTION("Bezier tet")
    {
        BezierSimplex<Scalar, 3, 2> bezier_tet;
        REQUIRE(bezier_tet.get_degree() == 2);
        REQUIRE(bezier_tet.get_simplex_dim() == 3);
        REQUIRE(bezier_tet.get_ordinate_dim() == -1);
        REQUIRE(bezier_tet.get_num_control_points() == 10);

        auto ctrl_pts = bezier_tet.get_control_points();

        Eigen::Matrix<Scalar, 10, 1> ordinates;
        ordinates << 0, 1, 1, 1, 2, 2, 2, 2, 2, 2;
        bezier_tet.set_ordinates(ordinates);

        for (Eigen::Index i = 0; i < 10; i++) {
            auto p = bezier_tet.evaluate(ctrl_pts.row(i));
            REQUIRE(p.isApprox(ordinates.row(i)));
        }

        Eigen::Matrix<Scalar, 1, 4> dir(0.3, 0.2, -0.5, 0.0);
        for (Eigen::Index i = 0; i < 10; i++) {
            Eigen::Matrix<Scalar, 1, 4> p = ctrl_pts.row(i);
            auto d_center = bezier_tet.evaluate_directional_derivative(p, dir);
            auto d_center_diff = finite_difference(bezier_tet, p, dir);
            REQUIRE((d_center - d_center_diff).norm() < 1e-4);
        }
    }
}
