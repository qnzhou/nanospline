#pragma once
#include <catch2/catch.hpp>
#include <limits>
#include <iostream>

namespace nanospline {

template<typename CurveType1, typename CurveType2,
    typename std::enable_if<std::is_same<
        typename CurveType1::Scalar,
        typename CurveType2::Scalar>::value,
        int>::type =0 >
void assert_same(const CurveType1& curve1, const CurveType2& curve2, int num_samples,
        const typename CurveType1::Scalar tol=1e-6) {
    using Scalar = typename CurveType1::Scalar;

    REQUIRE(curve1.get_domain_lower_bound() == Approx(curve2.get_domain_lower_bound()));
    REQUIRE(curve1.get_domain_upper_bound() == Approx(curve2.get_domain_upper_bound()));

    const Scalar t_min = std::max(
            curve1.get_domain_lower_bound(),
            curve2.get_domain_lower_bound());
    const Scalar t_max = std::min(
            curve1.get_domain_upper_bound(),
            curve2.get_domain_upper_bound());

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> samples;
    samples.setLinSpaced(num_samples+2, t_min, t_max);
    for (int i=0; i<num_samples+2; i++) {
        const auto p1 = curve1.evaluate(samples[i]);
        const auto p2 = curve2.evaluate(samples[i]);
        REQUIRE((p1-p2).norm() == Approx(0.0).margin(tol));
    }
}

template<typename CurveType1, typename CurveType2,
    typename std::enable_if<std::is_same<
        typename CurveType1::Scalar,
        typename CurveType2::Scalar>::value,
        int>::type =0 >
void assert_same(const CurveType1& curve1, const CurveType2& curve2, int num_samples,
        const typename CurveType1::Scalar lower_1,
        const typename CurveType1::Scalar upper_1,
        const typename CurveType1::Scalar lower_2,
        const typename CurveType1::Scalar upper_2,
        const typename CurveType1::Scalar tol=1e-6) {
    using Scalar = typename CurveType1::Scalar;

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> samples_1, samples_2;
    samples_1.setLinSpaced(num_samples+2, lower_1, upper_1);
    samples_2.setLinSpaced(num_samples+2, lower_2, upper_2);

    for (int i=0; i<num_samples+2; i++) {
        const auto p1 = curve1.evaluate(samples_1[i]);
        const auto p2 = curve2.evaluate(samples_2[i]);
        REQUIRE((p1-p2).norm() == Approx(0.0).margin(tol));
    }
}

/**
 * Validate derivative computation using finite difference.
 */
template<typename CurveType>
void validate_derivatives(const CurveType& curve, int num_samples,
        const typename CurveType::Scalar tol=1e-6) {
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
            REQUIRE(d[0]*delta == Approx(p1[0]-p0[0]).margin(tol));
            REQUIRE(d[1]*delta == Approx(p1[1]-p0[1]).margin(tol));
        } else if (i == num_samples+1) {
            t = t_max;
            auto p0 = curve.evaluate(t-delta);
            auto p1 = curve.evaluate(t);
            REQUIRE(d[0]*delta == Approx(p1[0]-p0[0]).margin(tol));
            REQUIRE(d[1]*delta == Approx(p1[1]-p0[1]).margin(tol));
        } else {
            // Center difference.
            const auto t0 = std::max(t_min, t-delta/2);
            const auto t1 = std::min(t_max, t+delta/2);
            const auto diff = t1-t0;
            auto p0 = curve.evaluate(t0);
            auto p1 = curve.evaluate(t1);
            //std::cout << d*diff << " : " << p1-p0 << std::endl;
            REQUIRE(d[0]*diff == Approx(p1[0]-p0[0]).margin(tol));
            REQUIRE(d[1]*diff == Approx(p1[1]-p0[1]).margin(tol));
        }
    }
}

/**
 * Validate 2nd derivative computation using finite difference.
 */
template<typename CurveType>
void validate_2nd_derivatives(const CurveType& curve, int num_samples,
        const typename CurveType::Scalar tol=1e-6) {
    using Scalar = typename CurveType::Scalar;

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> samples;
    const Scalar t_min = curve.get_domain_lower_bound();
    const Scalar t_max = curve.get_domain_upper_bound();
    samples.setLinSpaced(num_samples+2, t_min, t_max);
    constexpr Scalar delta = std::numeric_limits<Scalar>::epsilon() * 100;

    for (int i=0; i< num_samples+2; i++) {
        auto t = samples[i];
        auto d = curve.evaluate_2nd_derivative(t);

        if (i==0) {
            auto p0 = curve.evaluate_derivative(t_min);
            auto p1 = curve.evaluate_derivative(t_min+delta);
            REQUIRE(d[0]*delta == Approx(p1[0]-p0[0]).margin(tol));
            REQUIRE(d[1]*delta == Approx(p1[1]-p0[1]).margin(tol));
        } else if (i == num_samples+1) {
            t = t_max;
            auto p0 = curve.evaluate_derivative(t-delta);
            auto p1 = curve.evaluate_derivative(t);
            REQUIRE(d[0]*delta == Approx(p1[0]-p0[0]).margin(tol));
            REQUIRE(d[1]*delta == Approx(p1[1]-p0[1]).margin(tol));
        } else {
            // Center difference.
            const auto t0 = std::max(t_min, t-delta/2);
            const auto t1 = std::min(t_max, t+delta/2);
            const auto diff = t1-t0;
            auto p0 = curve.evaluate_derivative(t0);
            auto p1 = curve.evaluate_derivative(t1);
            REQUIRE(d[0]*diff == Approx(p1[0]-p0[0]).margin(tol));
            REQUIRE(d[1]*diff == Approx(p1[1]-p0[1]).margin(tol));
        }
    }
}

template<typename CurveType, typename CurveType2>
void validate_hodograph(const CurveType& curve, const CurveType2& hodograph,
        int num_samples, const typename CurveType::Scalar tol=1e-6) {
    using Scalar = typename CurveType::Scalar;

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> samples;
    const Scalar t_min = curve.get_domain_lower_bound();
    const Scalar t_max = curve.get_domain_upper_bound();
    samples.setLinSpaced(num_samples+2, t_min, t_max);

    for (int i=0; i< num_samples+2; i++) {
        auto t = samples[i];
        auto d = curve.evaluate_derivative(t);
        auto d2 = hodograph.evaluate(t);
        REQUIRE((d-d2).norm() == Approx(0.0).margin(tol));
    }
}


}
