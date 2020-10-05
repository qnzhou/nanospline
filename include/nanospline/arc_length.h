#pragma once

#include <nanospline/CurveBase.h>
#include <nanospline/Quadrature.h>

namespace nanospline {

/**
 * Given a parameter value t, this method computes the arc length at t using Romberg's method.
 */
template <typename Scalar, int DIM>
Scalar arc_length(const CurveBase<Scalar, DIM>& curve, Scalar t, size_t level = 10)
{
    auto speed = [&](Scalar tt) {
        const auto d1 = curve.evaluate_derivative(tt);
        return d1.norm();
    };

    constexpr Scalar tol = 1e-12;

    const Scalar t0 = curve.get_domain_lower_bound();
    std::vector<Scalar> R_values(level + 1, 0.0);
    Scalar h = t - t0;
    R_values[0] = 0.5 * h * (speed(t0), +speed(t));
    for (size_t i = 1; i <= level; i++) {
        h /= 2;
        Scalar R_above = R_values[0];

        R_values[0] = 0.5 * R_above;
        size_t num_steps = 1 << (i - 1);
        for (size_t k = 1; k < num_steps; k++) {
            R_values[0] += speed(t0 + (2 * k - 1) * h) * h;
        }

        size_t m = 1;
        Scalar R_prev_above = R_above;
        for (size_t j = 1; j <= i; j++) {
            m *= 4;
            R_above = R_values[j];
            R_values[j] = R_values[j - 1] + (R_values[j - 1] - R_prev_above) / (m - 1);
            R_prev_above = R_above;
        }

        if (i > 1 && std::abs(R_values[i] - R_values[i - 1]) < tol) {
            return R_values[i];
        }
    }

    return R_values[level];
}

/**
 * Given an arc length l, this method computes the parameter value t
 * corresponding to l.
 *
 * This method implements the following paper:
 *
 * Sharpe, Richard J., and Richard W. Thorne. "Numerical method for extracting an arc length
 * parameterization from parametric curves." Computer-aided design 14.2 (1982): 79-81.
 */
template <typename Scalar, int DIM>
Scalar inverse_arc_length(const CurveBase<Scalar, DIM>& curve, Scalar l, int num_iterations = 10)
{
    auto speed = [&](Scalar tt) {
        const auto d1 = curve.evaluate_derivative(tt);
        return d1.norm();
    };

    Scalar t = curve.get_domain_upper_bound();
    const Scalar L = arc_length(curve, t);
    t = l / L;

    const Scalar tol = L * 1e-12;

    for (int i = 0; i < num_iterations; i++) {
        const Scalar err = arc_length(curve, t) - l;
        if (std::abs(err) < tol) {
            break;
        }
        t = t - err / speed(t);
    }

    return t;
}

} // namespace nanospline
