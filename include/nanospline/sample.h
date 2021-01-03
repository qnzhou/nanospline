#pragma once

#include <nanospline/Exceptions.h>
#include <nanospline/arc_length.h>
#include <nanospline/enums.h>

#include <cassert>
#include <vector>

namespace nanospline {

namespace internal {

template <typename Scalar, int DIM>
void adaptive_sample(const CurveBase<Scalar, DIM>& curve,
    Scalar t0,
    Scalar t1,
    Scalar tol,
    std::vector<Scalar>& samples)
{
    const auto L = arc_length(curve, t0, t1);
    const auto p0 = curve.evaluate(t0);
    const auto p1 = curve.evaluate(t1);
    const auto l = (p1 - p0).norm();
    if ((L - l) > tol) {
        auto half = (t0 + t1) / 2;
        adaptive_sample(curve, t0, half, tol, samples);
        adaptive_sample(curve, half, t1, tol, samples);
    } else {
        samples.push_back(t0);
    }
}

} // namespace internal

/**
 * Sample the curve according to the sampling method.
 */
template <typename Scalar, int DIM>
std::vector<Scalar> sample(const CurveBase<Scalar, DIM>& curve,
    size_t num_samples,
    SampleMethod method = SampleMethod::UNIFORM_DOMAIN)
{
    assert(num_samples > 1);
    const Scalar min_t = curve.get_domain_lower_bound();
    const Scalar max_t = curve.get_domain_upper_bound();
    std::vector<Scalar> samples;
    samples.reserve(num_samples);

    switch (method) {
    case SampleMethod::UNIFORM_DOMAIN: {
        for (size_t i = 0; i < num_samples; i++) {
            samples.push_back(min_t + (max_t - min_t) * static_cast<Scalar>(i) /
                                          static_cast<Scalar>(num_samples - 1));
        }
    } break;
    case SampleMethod::UNIFORM_RANGE: {
        samples.push_back(min_t);
        const Scalar l = arc_length(curve, min_t, max_t);
        for (size_t i = 1; i < num_samples - 1; i++) {
            samples.push_back(inverse_arc_length(
                curve, l * static_cast<Scalar>(i) / static_cast<Scalar>(num_samples - 1)));
        }
        samples.push_back(max_t);
    } break;
    case SampleMethod::ADAPTIVE: {
        const auto l = arc_length(curve, min_t, max_t);
        const auto tol = (l / num_samples) / 10;
        internal::adaptive_sample(curve, min_t, max_t, tol, samples);
        samples.push_back(max_t);
    } break;
    default: throw not_implemented_error("Unsupported sampling method detected!"); break;
    }

    return samples;
}

} // namespace nanospline
