#pragma once
#include <catch2/catch.hpp>
#include <limits>

namespace nanospline {

/**
 * Validate derivative computation using finite difference.
 */
template<typename CurveType>
void validate_derivatives(const CurveType& curve, int num_samples) {
    using Scalar = typename CurveType::Scalar;

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> samples;
    const Scalar t_min = curve.get_domain_lower_bound();
    const Scalar t_max = curve.get_domain_upper_bound();
    samples.setLinSpaced(num_samples+2, t_min, t_max);
    constexpr Scalar delta = std::numeric_limits<Scalar>::epsilon() * 100;

    for (int i=0; i< num_samples+2; i++) {
        auto t = samples[i];
        auto d = curve.evaluate_derivative(t);

        if (i==0) {
            auto p0 = curve.evaluate(t_min);
            auto p1 = curve.evaluate(t_min+delta);
            REQUIRE(d[0]*delta == Approx(p1[0]-p0[0]).margin(1e-6));
            REQUIRE(d[1]*delta == Approx(p1[1]-p0[1]).margin(1e-6));
        } else if (i == num_samples+1) {
            t = t_max;
            auto p0 = curve.evaluate(t-delta);
            auto p1 = curve.evaluate(t);
            REQUIRE(d[0]*delta == Approx(p1[0]-p0[0]).margin(1e-6));
            REQUIRE(d[1]*delta == Approx(p1[1]-p0[1]).margin(1e-6));
        } else {
            // Center difference.
            const auto t0 = std::max(t_min, t-delta/2);
            const auto t1 = std::max(t_min, t-delta/2);
            const auto diff = t1-t0;
            auto p0 = curve.evaluate(t0);
            auto p1 = curve.evaluate(t1);
            REQUIRE(d[0]*diff == Approx(p1[0]-p0[0]).margin(1e-6));
            REQUIRE(d[1]*diff == Approx(p1[1]-p0[1]).margin(1e-6));
        }
    }
}

}
