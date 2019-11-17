#include <catch2/catch.hpp>

#include <nanospline/Bezier.h>

TEST_CASE("Bezier", "[bezier]") {
    using namespace nanospline;

    SECTION("Generic degree 0") {
        Eigen::Matrix<float, 1, 2> control_pts;
        control_pts << 0.0, 0.1;
        Bezier<float, 2, 0, true> curve;
        curve.set_control_points(control_pts);

        auto start = curve.evaluate(0);
        auto mid = curve.evaluate(0.5);
        auto end = curve.evaluate(1);

        REQUIRE((start-control_pts.row(0)).norm() == Approx(0.0));
        REQUIRE((end-control_pts.row(0)).norm() == Approx(0.0));
        REQUIRE((mid-control_pts.row(0)).norm() == Approx(0.0));
    }

    SECTION("Generic degree 1") {
        Eigen::Matrix<float, 2, 2> control_pts;
        control_pts << 0.0, 0.0,
                       1.0, 0.0;
        Bezier<float, 2, 1, true> curve;
        curve.set_control_points(control_pts);

        auto start = curve.evaluate(0);
        auto mid = curve.evaluate(0.5);
        auto end = curve.evaluate(1);

        REQUIRE(start[0] == Approx(0.0));
        REQUIRE(mid[0] == Approx(0.5));
        REQUIRE(end[0] == Approx(1.0));

        REQUIRE(start[1] == Approx(0.0));
        REQUIRE(mid[1] == Approx(0.0));
        REQUIRE(end[1] == Approx(0.0));
    }

    SECTION("Generic degree 3") {
        Eigen::Matrix<float, 4, 2> control_pts;
        control_pts << 0.0, 0.0,
                       1.0, 1.0,
                       2.0, 1.0,
                       3.0, 0.0;
        Bezier<float, 2, 3, true> curve;
        curve.set_control_points(control_pts);

        SECTION("Ends") {
            auto start = curve.evaluate(0);
            auto end = curve.evaluate(1);

            REQUIRE((start-control_pts.row(0)).norm() == Approx(0.0));
            REQUIRE((end-control_pts.row(3)).norm() == Approx(0.0));
        }

        SECTION("Mid point") {
            auto p = curve.evaluate(0.5);
            REQUIRE(p[0] == Approx(1.5));
            REQUIRE(p[1] > 0.0);
            REQUIRE(p[1] < 1.0);
        }

        SECTION("Inverse evaluation") {
            Eigen::Matrix<float, 1, 2> p(0.0, 1.0);
            REQUIRE_THROWS(curve.inverse_evaluate(p));
        }

        SECTION("Approximate inverse evaluation") {
            auto t = curve.approximate_inverse_evaluate({1.5, 1.1});
            REQUIRE(t == Approx(0.5));

            t = curve.approximate_inverse_evaluate({0.0, -1.0});
            REQUIRE(t == Approx(0.0));

            t = curve.approximate_inverse_evaluate({3.1, 0.0});
            REQUIRE(t == Approx(1.0));
        }
    }

    SECTION("Dynmaic degree") {
        Eigen::Matrix<float, 4, 2> control_pts;
        control_pts << 0.0, 0.0,
                       1.0, 1.0,
                       2.0, 1.0,
                       3.0, 0.0;
        Bezier<float, 2, -1> curve;
        curve.set_control_points(control_pts);

        SECTION("Ends") {
            auto start = curve.evaluate(0);
            auto end = curve.evaluate(1);

            REQUIRE((start-control_pts.row(0)).norm() == Approx(0.0));
            REQUIRE((end-control_pts.row(3)).norm() == Approx(0.0));
        }

        SECTION("Mid point") {
            auto p = curve.evaluate(0.5);
            REQUIRE(p[0] == Approx(1.5));
            REQUIRE(p[1] > 0.0);
            REQUIRE(p[1] < 1.0);
        }

        SECTION("Inverse evaluation") {
            Eigen::Matrix<float, 1, 2> p(0.0, 1.0);
            REQUIRE_THROWS(curve.inverse_evaluate(p));
        }

        SECTION("Approximate inverse evaluation") {
            auto t = curve.approximate_inverse_evaluate({1.5, 1.1});
            REQUIRE(t == Approx(0.5));

            t = curve.approximate_inverse_evaluate({0.0, -1.0});
            REQUIRE(t == Approx(0.0));

            t = curve.approximate_inverse_evaluate({3.1, 0.0});
            REQUIRE(t == Approx(1.0));
        }
    }

    SECTION("Specialized degree 0") {
        Eigen::Matrix<float, 1, 2> control_pts;
        control_pts << 0.0, 0.1;
        Bezier<float, 2, 0> curve;
        curve.set_control_points(control_pts);

        Eigen::Matrix<float, 1, 2> p(0.0, 1.0);
        curve.inverse_evaluate(p);

        SECTION("Consistency") {
            Bezier<float, 2, 0, true> generic_curve;
            generic_curve.set_control_points(control_pts);
            constexpr int N=10;
            for (int i=0; i<N; i++) {
                float t = float(i) / float(N);
                const auto p = curve.evaluate(t);
                const auto q = generic_curve.evaluate(t);
                REQUIRE(p[0] == Approx(q[0]));
                REQUIRE(p[1] == Approx(q[1]));
            }
        }

    }

    SECTION("Specialized degree 1") {
        Eigen::Matrix<float, 2, 2> control_pts;
        control_pts << 0.0, 0.0,
                       1.0, 1.0;
        Bezier<float, 2, 1> curve;
        curve.set_control_points(control_pts);

        SECTION("Consistency") {
            Bezier<float, 2, 1, true> generic_curve;
            generic_curve.set_control_points(control_pts);
            constexpr int N=10;
            for (int i=0; i<N; i++) {
                float t = float(i) / float(N);
                const auto p = curve.evaluate(t);
                const auto q = generic_curve.evaluate(t);
                REQUIRE(p[0] == Approx(q[0]));
                REQUIRE(p[1] == Approx(q[1]));
            }
        }

        SECTION("Evaluation") {
            auto start = curve.evaluate(0.0);
            auto mid = curve.evaluate(0.5);
            auto end = curve.evaluate(1.0);

            REQUIRE((start - control_pts.row(0)).norm() == Approx(0.0));
            REQUIRE((end - control_pts.row(1)).norm() == Approx(0.0));
            REQUIRE(mid[0] == Approx(0.5));
            REQUIRE(mid[1] == Approx(0.5));
        }

        SECTION("Inverse evaluate") {
            float t0 = 0.2f;
            const auto p0 = curve.evaluate(t0);
            const auto t = curve.inverse_evaluate(p0);
            REQUIRE(t0 == Approx(t));

            Eigen::Matrix<float, 1, 2> p1(1.0, 0.0);
            const auto t1 = curve.inverse_evaluate(p1);
            REQUIRE(t1 == Approx(0.5));

            Eigen::Matrix<float, 1, 2> p2(-1.0, 0.0);
            const auto t2 = curve.inverse_evaluate(p2);
            REQUIRE(t2 == Approx(0.0));

            Eigen::Matrix<float, 1, 2> p3(1.0, 1.1);
            const auto t3 = curve.inverse_evaluate(p3);
            REQUIRE(t3 == Approx(1.0));
        }
    }

    SECTION("Specialized degree 2") {
        Eigen::Matrix<float, 3, 2> control_pts;
        control_pts << 0.0, 0.0,
                       1.0, 1.0,
                       2.0, 0.0;
        Bezier<float, 2, 2> curve;
        curve.set_control_points(control_pts);

        SECTION("Consistency") {
            Bezier<float, 2, 2, true> generic_curve;
            generic_curve.set_control_points(control_pts);
            constexpr int N=10;
            for (int i=0; i<N; i++) {
                float t = float(i) / float(N);
                const auto p = curve.evaluate(t);
                const auto q = generic_curve.evaluate(t);
                REQUIRE(p[0] == Approx(q[0]));
                REQUIRE(p[1] == Approx(q[1]));
            }
        }

        SECTION("Evaluation") {
            const auto start = curve.evaluate(0.0);
            const auto mid = curve.evaluate(0.5);
            const auto end = curve.evaluate(1.0);

            REQUIRE((start-control_pts.row(0)).norm() == Approx(0.0));
            REQUIRE((end-control_pts.row(2)).norm() == Approx(0.0));
            REQUIRE(mid[0] == Approx(1.0));
            REQUIRE(mid[1] < 1.0);
        }
    }

    SECTION("Specialized degree 3") {
        Eigen::Matrix<float, 4, 2> control_pts;
        control_pts << 0.0, 0.0,
                       1.0, 1.0,
                       2.0, 1.0,
                       3.0, 0.0;
        Bezier<float, 2, 3> curve;
        curve.set_control_points(control_pts);

        SECTION("Consistency") {
            Bezier<float, 2, 3, true> generic_curve;
            generic_curve.set_control_points(control_pts);
            constexpr int N=10;
            for (int i=0; i<N; i++) {
                float t = float(i) / float(N);
                const auto p = curve.evaluate(t);
                const auto q = generic_curve.evaluate(t);
                REQUIRE(p[0] == Approx(q[0]));
                REQUIRE(p[1] == Approx(q[1]));
            }
        }

        SECTION("Evaluation") {
            const auto start = curve.evaluate(0.0);
            const auto mid = curve.evaluate(0.5);
            const auto end = curve.evaluate(1.0);

            REQUIRE((start-control_pts.row(0)).norm() == Approx(0.0));
            REQUIRE((end-control_pts.row(3)).norm() == Approx(0.0));
            REQUIRE(mid[0] == Approx(1.5));
            REQUIRE(mid[1] < 1.0);
        }
    }

}
