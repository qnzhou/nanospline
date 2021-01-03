#include <catch2/catch.hpp>

#include <nanospline/sample.h>

TEST_CASE("sample", "[sample]") {
    using namespace nanospline;
    using Scalar = double;


    SECTION("Linear") {
        Eigen::Matrix<Scalar, 2, 2> ctrl_pts;
        ctrl_pts << 0.0, 0.0,
                    1.0, 1.0;
        Bezier<Scalar, 2, 1> curve;
        curve.set_control_points(ctrl_pts);

        auto samples = sample(curve, 3);
        REQUIRE(samples.size() == 3);
        REQUIRE(samples.front() == 0);
        REQUIRE(samples.back() == 1);
        REQUIRE(samples[1] == 0.5);

        samples = sample(curve, 3, SampleMethod::UNIFORM_RANGE);
        REQUIRE(samples.size() == 3);
        REQUIRE(samples.front() == 0);
        REQUIRE(samples.back() == 1);
        REQUIRE(samples[1] == 0.5);
    }

    SECTION("Non-linear but symmetric") {
        Eigen::Matrix<Scalar, 3, 2> ctrl_pts;
        ctrl_pts << 0.0, 0.0,
                    1.0, 0.0,
                    1.0, 1.0;
        Bezier<Scalar, 2, 2> curve;
        curve.set_control_points(ctrl_pts);

        auto samples = sample(curve, 3);
        REQUIRE(samples.size() == 3);
        REQUIRE(samples.front() == 0);
        REQUIRE(samples.back() == 1);
        REQUIRE(samples[1] == 0.5);

        samples = sample(curve, 3, SampleMethod::UNIFORM_RANGE);
        REQUIRE(samples.size() == 3);
        REQUIRE(samples.front() == 0);
        REQUIRE(samples.back() == 1);
        REQUIRE(samples[1] == Approx(0.5).epsilon(1e-2));
    }

    SECTION("Circle")
    {
        constexpr Scalar R = 12.1;
        Eigen::Matrix<Scalar, 1, 2> c(0.0, R);
        Eigen::Matrix<Scalar, 9, 2> control_pts;
        control_pts << 0.0, 0.0, R, 0.0, R, R, R, 2 * R, 0.0, 2 * R, -R, 2 * R, -R, R, -R, 0.0, 0.0,
            0.0;
        Eigen::Matrix<Scalar, 12, 1> knots;
        knots << 0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0;
        Eigen::Matrix<Scalar, 9, 1> weights;
        weights << 1.0, sqrt(2) / 2, 1.0, sqrt(2) / 2, 1.0, sqrt(2) / 2, 1.0, sqrt(2) / 2, 1.0;

        NURBS<Scalar, 2, 2, true> curve;
        curve.set_control_points(control_pts);
        curve.set_knots(knots);
        curve.set_weights(weights);
        curve.initialize();

        SECTION("UNIFORM_RANGE") {
            constexpr size_t N = 10;
            auto samples = sample(curve, N, SampleMethod::UNIFORM_RANGE);
            REQUIRE(samples.size() == N);

            const Scalar L = arc_length(curve, samples[0], samples[N-1]);
            REQUIRE(L == Approx(M_PI*R*2).epsilon(1e-2));
            for (size_t i=1; i<N; i++) {
                const Scalar l = arc_length(curve, samples[i-1], samples[i]);
                REQUIRE(l == Approx(M_PI*R*2/(N-1)).epsilon(2e-2));
            }
        }
        SECTION("Adaptive") {
            constexpr size_t N = 10;
            auto samples = sample(curve, N, SampleMethod::ADAPTIVE);
            for (size_t i=1; i<samples.size(); i++) {
                const Scalar l = arc_length(curve, samples[i-1], samples[i]);
                REQUIRE(l == Approx(M_PI*R*2/(samples.size()-1)).epsilon(2e-2));
            }
        }
    }
}
