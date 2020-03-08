#pragma once
#include <catch2/catch.hpp>
#include <limits>
#include <iostream>
#include <nanospline/hodograph.h>

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
    const Scalar delta = (t_max - t_min) * 1e-6;

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

template<typename PatchType>
void validate_derivative(const PatchType& patch, int u_samples, int v_samples,
        const typename PatchType::Scalar tol=1e-6) {
    const auto u_min = patch.get_u_lower_bound();
    const auto u_max = patch.get_u_upper_bound();
    const auto v_min = patch.get_v_lower_bound();
    const auto v_max = patch.get_v_upper_bound();

    const auto delta_u = (u_max-u_min) * 1e-6;
    const auto delta_v = (v_max-v_min) * 1e-6;

    for (int i=0; i<=u_samples; i++) {
        const auto u = i * (u_max-u_min) / u_samples + u_min;
        for (int j=0; j<=v_samples; j++) {
            const auto v = j * (v_max-v_min) / v_samples + v_min;

            auto du = patch.evaluate_derivative_u(u, v);
            auto dv = patch.evaluate_derivative_v(u, v);

            // Center difference.
            const auto u_prev = std::max(u-delta_u, u_min);
            const auto u_next = std::min(u+delta_u, u_max);
            const auto v_prev = std::max(v-delta_v, v_min);
            const auto v_next = std::min(v+delta_v, v_max);

            const auto p_u_prev = patch.evaluate(u_prev, v);
            const auto p_u_next = patch.evaluate(u_next, v);
            const auto p_v_prev = patch.evaluate(u, v_prev);
            const auto p_v_next = patch.evaluate(u, v_next);

            REQUIRE(du[0]*(u_next-u_prev) == Approx(p_u_next[0]-p_u_prev[0]).margin(tol));
            REQUIRE(du[1]*(u_next-u_prev) == Approx(p_u_next[1]-p_u_prev[1]).margin(tol));
            REQUIRE(dv[0]*(v_next-v_prev) == Approx(p_v_next[0]-p_v_prev[0]).margin(tol));
            REQUIRE(dv[1]*(v_next-v_prev) == Approx(p_v_next[1]-p_v_prev[1]).margin(tol));
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
    //constexpr Scalar delta = std::numeric_limits<Scalar>::epsilon() * 1e3;
    const Scalar delta = (t_max - t_min) * 1e-6;

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

/**
 * Validate 2nd derivative computation using hodograph.
 */
template<typename Scalar, int dim, int degree, bool generic>
void validate_2nd_derivatives(const Bezier<Scalar, dim, degree, generic>& curve,
        int num_samples, const Scalar tol=1e-6) {

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> samples;
    const Scalar t_min = curve.get_domain_lower_bound();
    const Scalar t_max = curve.get_domain_upper_bound();
    samples.setLinSpaced(num_samples+2, t_min, t_max);

    auto hodograph = compute_hodograph(curve);
    auto hodograph2 = compute_hodograph(hodograph);

    for (int i=0; i< num_samples+2; i++) {
        auto t = samples[i];
        auto d = curve.evaluate_2nd_derivative(t);
        auto c = hodograph2.evaluate(t);
        REQUIRE((d-c).norm() == Approx(0.0).margin(tol));
    }
}

/**
 * Validate 2nd derivative computation using hodograph.
 */
template<typename Scalar, int dim, int degree, bool generic>
void validate_2nd_derivatives(const BSpline<Scalar, dim, degree, generic>& curve,
        int num_samples, const Scalar tol=1e-6) {

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> samples;
    const Scalar t_min = curve.get_domain_lower_bound();
    const Scalar t_max = curve.get_domain_upper_bound();
    samples.setLinSpaced(num_samples+2, t_min, t_max);

    auto hodograph = compute_hodograph(curve);
    auto hodograph2 = compute_hodograph(hodograph);

    for (int i=0; i< num_samples+2; i++) {
        auto t = samples[i];
        auto d = curve.evaluate_2nd_derivative(t);
        auto c = hodograph2.evaluate(t);
        REQUIRE((d-c).norm() == Approx(0.0).margin(tol));
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

template<typename PatchType>
void validate_iso_curves(const PatchType& patch, int num_samples=10) {
    const auto u_min = patch.get_u_lower_bound();
    const auto u_max = patch.get_u_upper_bound();
    const auto v_min = patch.get_v_lower_bound();
    const auto v_max = patch.get_v_upper_bound();

    for (int i=0; i<=num_samples; i++) {
        for (int j=0; j<=num_samples; j++) {
            const auto u = i * (u_max-u_min) / (num_samples) + u_min;
            const auto v = j * (v_max-v_min) / (num_samples) + v_min;
            auto u_curve = patch.compute_iso_curve_u(v);
            auto v_curve = patch.compute_iso_curve_v(u);
            auto p = patch.evaluate(u, v);
            auto p1 = u_curve.evaluate(u);
            auto p2 = v_curve.evaluate(v);
            REQUIRE((p-p1).norm() == Approx(0.0).margin(1e-6));
            REQUIRE((p-p2).norm() == Approx(0.0).margin(1e-6));
        }
    }
}


}
