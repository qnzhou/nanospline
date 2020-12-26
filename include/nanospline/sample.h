#pragma once

#include <nanospline/Exceptions.h>
#include <nanospline/arc_length.h>
#include <nanospline/enums.h>

#include <cassert>
#include <vector>

namespace nanospline {

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
    std::vector<Scalar> samples(num_samples);

    switch (method) {
    case SampleMethod::UNIFORM_DOMAIN: {
        for (size_t i = 0; i < num_samples; i++) {
            samples[i] = min_t + (max_t - min_t) * i / (num_samples - 1);
        }
    } break;
    case SampleMethod::UNIFORM_RANGE: {
        samples[0] = min_t;
        samples[num_samples - 1] = max_t;
        const Scalar l = arc_length(curve, min_t, max_t);
        for (size_t i = 1; i < num_samples - 1; i++) {
            samples[i] = inverse_arc_length(curve, l * i / (num_samples - 1));
        }
    } break;
    default: throw not_implemented_error("Unsupported sampling method detected!"); break;
    }

    return samples;
}

} // namespace nanospline
