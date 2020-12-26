#pragma once

#include <nanospline/CurveBase.h>
#include <nanospline/PatchBase.h>

namespace nanospline {

namespace internal {

/**
 * This function implement Romberg's method to integrate speed function from t0
 * to t1.
 */
template <typename Scalar>
Scalar romberg(std::function<Scalar(Scalar)> speed, Scalar t0, Scalar t1, size_t level, Scalar tol);

} // namespace internal

/**
 * Compute the arc length of the input curve between t0 and t1.
 */
template <typename Scalar, int DIM>
Scalar arc_length(const CurveBase<Scalar, DIM>& curve, Scalar t0, Scalar t1, size_t level = 10, Scalar tol=1e-12)
{
    std::function<Scalar(Scalar)> speed = [&](Scalar tt) -> Scalar {
        const auto d1 = curve.evaluate_derivative(tt);
        return d1.norm();
    };

    return internal::romberg(speed, t0, t1, level, tol);
}

/**
 * Given a parameter value t, this method computes the arc length at t using Romberg's method.
 */
template <typename Scalar, int DIM>
Scalar arc_length(const CurveBase<Scalar, DIM>& curve, Scalar t, size_t level = 10, Scalar tol=1e-12)
{
    const Scalar t0 = curve.get_domain_lower_bound();
    return arc_length(curve, t0, t, level, tol);
}

/**
 * Compute arc length between two points on a patch.
 */
template <typename Scalar, int DIM>
Scalar arc_length(const PatchBase<Scalar, DIM>& patch,
    Scalar u0,
    Scalar v0,
    Scalar u1,
    Scalar v1,
    size_t level = 10,
    Scalar tol = 1e-12)
{
    const Eigen::Matrix<Scalar, 1, 2> dir = {u1 - u0, v1 - v0};

    auto position = [&](Scalar t) {
        return Eigen::Matrix<Scalar, 1, 2>((u1 - u0) * t + u0, (v1 - v0) * t + v0);
    };

    std::function<Scalar(Scalar)> speed = [&](Scalar t) -> Scalar {
        const auto p = position(t);
        const auto du = patch.evaluate_derivative_u(p[0], p[1]);
        const auto dv = patch.evaluate_derivative_v(p[0], p[1]);
        return (du * dir[0] + dv * dir[1]).norm();
    };

    return internal::romberg(speed, (Scalar)0, (Scalar)1, level, tol);
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

template <typename Scalar, int DIM>
Eigen::Matrix<Scalar, 1, 2> inverse_arc_length(const PatchBase<Scalar, DIM>& patch,
    Scalar u0,
    Scalar v0,
    Scalar u1,
    Scalar v1,
    Scalar l,
    int num_iterations = 10)
{
    const Eigen::Matrix<Scalar, 1, 2> dir = {u1 - u0, v1 - v0};

    auto position = [&](Scalar t) {
        return Eigen::Matrix<Scalar, 1, 2>((u1 - u0) * t + u0, (v1 - v0) * t + v0);
    };

    std::function<Scalar(Scalar)> speed = [&](Scalar t) -> Scalar {
        const auto p = position(t);
        const auto du = patch.evaluate_derivative_u(p[0], p[1]);
        const auto dv = patch.evaluate_derivative_v(p[0], p[1]);
        return (du * dir[0] + dv * dir[1]).norm();
    };

    const auto L = arc_length(patch, u0, v0, u1, v1);
    Scalar t = l / L;

    const Scalar tol = L * 1e-12;

    for (int i = 0; i < num_iterations; i++) {
        const auto p = position(t);
        const Scalar err = arc_length(patch, u0, v0, p[0], p[1]) - l;
        if (std::abs(err) < tol) {
            break;
        }
        t = t - err / speed(t);
    }

    return position(t);
}

namespace internal {

template <typename Scalar>
Scalar romberg(std::function<Scalar(Scalar)> speed, Scalar t0, Scalar t1, size_t level, Scalar tol)
{
    std::vector<Scalar> R_values(level + 1, 0.0);
    Scalar h = t1 - t0;
    R_values[0] = 0.5 * h * (speed(t0) + speed(t1));
    for (size_t i = 1; i <= level; i++) {
        h /= 2;
        Scalar R_above = R_values[0];

        R_values[0] = 0.5 * R_above;
        size_t num_steps = (size_t)1 << (i - 1);
        for (size_t k = 1; k < num_steps; k++) {
            R_values[0] += speed(t0 + static_cast<Scalar>(2 * k - 1) * h) * h;
        }

        size_t m = 1;
        Scalar R_prev_above = R_above;
        for (size_t j = 1; j <= i; j++) {
            m *= 4;
            R_above = R_values[j];
            R_values[j] =
                R_values[j - 1] + (R_values[j - 1] - R_prev_above) / static_cast<Scalar>(m - 1);
            R_prev_above = R_above;
        }

        if (i > 1 && std::abs(R_values[i] - R_values[i - 1]) < tol) {
            return R_values[i];
        }
    }

    return R_values[level];
}

} // namespace internal
} // namespace nanospline
