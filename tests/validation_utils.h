#pragma once
#include <nanospline/Line.h>
#include <nanospline/PatchBase.h>
#include <nanospline/hodograph.h>
#include <nanospline/save_msh.h>
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <limits>

namespace nanospline {

template <typename CurveType1,
    typename CurveType2,
    typename std::enable_if<
        std::is_same<typename CurveType1::Scalar, typename CurveType2::Scalar>::value,
        int>::type = 0>
void assert_same(const CurveType1& curve1,
    const CurveType2& curve2,
    int num_samples,
    const typename CurveType1::Scalar tol = 1e-6)
{
    using Scalar = typename CurveType1::Scalar;

    REQUIRE_THAT(curve1.get_domain_lower_bound(),
        Catch::Matchers::WithinAbs(curve2.get_domain_lower_bound(), 1e-6));
    REQUIRE_THAT(curve1.get_domain_upper_bound(),
        Catch::Matchers::WithinAbs(curve2.get_domain_upper_bound(), 1e-6));

    const Scalar t_min = std::max(curve1.get_domain_lower_bound(), curve2.get_domain_lower_bound());
    const Scalar t_max = std::min(curve1.get_domain_upper_bound(), curve2.get_domain_upper_bound());

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> samples;
    samples.setLinSpaced(num_samples + 2, t_min, t_max);
    for (int i = 0; i < num_samples + 2; i++) {
        const auto p1 = curve1.evaluate(samples[i]);
        const auto p2 = curve2.evaluate(samples[i]);
        REQUIRE_THAT((p1 - p2).norm(), Catch::Matchers::WithinAbs(0.0, tol));
    }
}

template <typename CurveType1,
    typename CurveType2,
    typename std::enable_if<
        std::is_same<typename CurveType1::Scalar, typename CurveType2::Scalar>::value,
        int>::type = 0>
void assert_same(const CurveType1& curve1,
    const CurveType2& curve2,
    int num_samples,
    const typename CurveType1::Scalar lower_1,
    const typename CurveType1::Scalar upper_1,
    const typename CurveType1::Scalar lower_2,
    const typename CurveType1::Scalar upper_2,
    const typename CurveType1::Scalar tol = 1e-6)
{
    using Scalar = typename CurveType1::Scalar;

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> samples_1, samples_2;
    samples_1.setLinSpaced(num_samples + 2, lower_1, upper_1);
    samples_2.setLinSpaced(num_samples + 2, lower_2, upper_2);

    for (int i = 0; i < num_samples + 2; i++) {
        const auto p1 = curve1.evaluate(samples_1[i]);
        const auto p2 = curve2.evaluate(samples_2[i]);
        REQUIRE_THAT((p1 - p2).norm(), Catch::Matchers::WithinAbs(0.0, tol));
    }
}

template <typename PatchType1,
    typename PatchType2,
    typename std::enable_if<
        std::is_same<typename PatchType1::Scalar, typename PatchType2::Scalar>::value,
        int>::type = 0>
void assert_same(const PatchType1& patch1,
    const PatchType2& patch2,
    int num_samples,
    const typename PatchType1::Scalar u_min_1,
    const typename PatchType1::Scalar u_max_1,
    const typename PatchType1::Scalar v_min_1,
    const typename PatchType1::Scalar v_max_1,
    const typename PatchType1::Scalar u_min_2,
    const typename PatchType1::Scalar u_max_2,
    const typename PatchType1::Scalar v_min_2,
    const typename PatchType1::Scalar v_max_2,
    const typename PatchType1::Scalar tol = 1e-6)
{
    using Scalar = typename PatchType1::Scalar;

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> samples_v1, samples_v2, samples_u1, samples_u2;
    samples_u1.setLinSpaced(num_samples + 2, u_min_1, u_max_1);
    samples_v1.setLinSpaced(num_samples + 2, v_min_1, v_max_1);
    samples_u2.setLinSpaced(num_samples + 2, u_min_2, u_max_2);
    samples_v2.setLinSpaced(num_samples + 2, v_min_2, v_max_2);
    for (int i = 0; i < num_samples + 2; i++) {
        for (int j = 0; j < num_samples + 2; j++) {
            const auto p1 = patch1.evaluate(samples_u1[i], samples_v1[j]);
            const auto p2 = patch2.evaluate(samples_u2[i], samples_v2[j]);
            REQUIRE_THAT((p1 - p2).norm(), Catch::Matchers::WithinAbs(0.0, tol));
        }
    }
}


/**
 * Validate derivative computation using finite difference.
 */
template <typename CurveType>
void validate_derivatives(
    const CurveType& curve, int num_samples, const typename CurveType::Scalar tol = 1e-6)
{
    using Scalar = typename CurveType::Scalar;

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> samples;
    const Scalar t_min = curve.get_domain_lower_bound();
    const Scalar t_max = curve.get_domain_upper_bound();
    samples.setLinSpaced(num_samples + 2, t_min, t_max);
    const Scalar delta = (t_max - t_min) * 1e-6;

    for (int i = 0; i < num_samples + 2; i++) {
        auto t = samples[i];
        auto d = curve.evaluate_derivative(t);

        if (i == 0) {
            auto p0 = curve.evaluate(t_min);
            auto p1 = curve.evaluate(t_min + delta);
            REQUIRE_THAT(d[0] * delta, Catch::Matchers::WithinAbs(p1[0] - p0[0], tol));
            REQUIRE_THAT(d[1] * delta, Catch::Matchers::WithinAbs(p1[1] - p0[1], tol));
        } else if (i == num_samples + 1) {
            t = t_max;
            auto p0 = curve.evaluate(t - delta);
            auto p1 = curve.evaluate(t);
            REQUIRE_THAT(d[0] * delta, Catch::Matchers::WithinAbs(p1[0] - p0[0], tol));
            REQUIRE_THAT(d[1] * delta, Catch::Matchers::WithinAbs(p1[1] - p0[1], tol));
        } else {
            // Center difference.
            const auto t0 = std::max(t_min, t - delta / 2);
            const auto t1 = std::min(t_max, t + delta / 2);
            const auto diff = t1 - t0;
            auto p0 = curve.evaluate(t0);
            auto p1 = curve.evaluate(t1);
            REQUIRE_THAT(d[0] * diff, Catch::Matchers::WithinAbs(p1[0] - p0[0], tol));
            REQUIRE_THAT(d[1] * diff, Catch::Matchers::WithinAbs(p1[1] - p0[1], tol));
        }
    }
}

template <typename PatchType>
void validate_derivative(const PatchType& patch,
    int u_samples,
    int v_samples,
    const typename PatchType::Scalar tol = 1e-6,
    bool test_2nd_derivatives = true)
{
    const auto dim = patch.get_dim();
    const auto u_min = patch.get_u_lower_bound();
    const auto u_max = patch.get_u_upper_bound();
    const auto v_min = patch.get_v_lower_bound();
    const auto v_max = patch.get_v_upper_bound();

    const auto delta_u = (u_max - u_min) * 1e-6;
    const auto delta_v = (v_max - v_min) * 1e-6;

    for (int i = 0; i <= u_samples; i++) {
        const auto u = i * (u_max - u_min) / u_samples + u_min;
        for (int j = 0; j <= v_samples; j++) {
            const auto v = j * (v_max - v_min) / v_samples + v_min;

            const auto du = patch.evaluate_derivative_u(u, v);
            const auto dv = patch.evaluate_derivative_v(u, v);

            // Center difference.
            const auto u_prev = std::max(u - delta_u, u_min);
            const auto u_next = std::min(u + delta_u, u_max);
            const auto v_prev = std::max(v - delta_v, v_min);
            const auto v_next = std::min(v + delta_v, v_max);

            const auto p_u_prev = patch.evaluate(u_prev, v);
            const auto p_u_next = patch.evaluate(u_next, v);
            const auto p_v_prev = patch.evaluate(u, v_prev);
            const auto p_v_next = patch.evaluate(u, v_next);

            for (int k = 0; k < dim; k++) {
                REQUIRE_THAT(du[k] * (u_next - u_prev),
                    Catch::Matchers::WithinAbs(p_u_next[k] - p_u_prev[k], tol));
                REQUIRE_THAT(dv[k] * (v_next - v_prev),
                    Catch::Matchers::WithinAbs(p_v_next[k] - p_v_prev[k], tol));
            }

            if (test_2nd_derivatives) {
                const auto duu = patch.evaluate_2nd_derivative_uu(u, v);
                const auto dvv = patch.evaluate_2nd_derivative_vv(u, v);

                const auto du_prev = patch.evaluate_derivative_u(u_prev, v);
                const auto du_next = patch.evaluate_derivative_u(u_next, v);
                const auto dv_prev = patch.evaluate_derivative_v(u, v_prev);
                const auto dv_next = patch.evaluate_derivative_v(u, v_next);

                for (int k = 0; k < dim; k++) {
                    REQUIRE_THAT(duu[k] * (u_next - u_prev),
                        Catch::Matchers::WithinAbs(du_next[k] - du_prev[k], tol));
                    REQUIRE_THAT(dvv[k] * (v_next - v_prev),
                        Catch::Matchers::WithinAbs(dv_next[k] - dv_prev[k], tol));
                }

                const auto duv = patch.evaluate_2nd_derivative_uv(u, v);

                const auto du_v_prev = patch.evaluate_derivative_u(u, v_prev);
                const auto du_v_next = patch.evaluate_derivative_u(u, v_next);
                const auto dv_u_prev = patch.evaluate_derivative_v(u_prev, v);
                const auto dv_u_next = patch.evaluate_derivative_v(u_next, v);

                for (int k = 0; k < dim; k++) {
                    REQUIRE_THAT(duv[k] * (v_next - v_prev),
                        Catch::Matchers::WithinAbs(du_v_next[k] - du_v_prev[k], tol));
                    REQUIRE_THAT(duv[k] * (u_next - u_prev),
                        Catch::Matchers::WithinAbs(dv_u_next[k] - dv_u_prev[k], tol));
                }
            }
        }
    }
}

template <typename PatchType>
void validate_derivative_patches(const PatchType& patch,
    int u_samples,
    int v_samples,
    const typename PatchType::Scalar tol = 1e-6)
{
    const auto u_min = patch.get_u_lower_bound();
    const auto u_max = patch.get_u_upper_bound();
    const auto v_min = patch.get_v_lower_bound();
    const auto v_max = patch.get_v_upper_bound();

    const auto du_patch = patch.compute_du_patch();
    const auto dv_patch = patch.compute_dv_patch();

    // Three ways of computing duv.
    const auto duv_patch = du_patch.compute_dv_patch();
    const auto dvu_patch = dv_patch.compute_du_patch();
    const auto duv_patch_explicit = patch.compute_duv_patch();

    for (int i = 0; i <= u_samples; i++) {
        const auto u = i * (u_max - u_min) / u_samples + u_min;
        for (int j = 0; j <= v_samples; j++) {
            const auto v = j * (v_max - v_min) / v_samples + v_min;

            auto du = patch.evaluate_derivative_u(u, v);
            auto dv = patch.evaluate_derivative_v(u, v);

            auto du_p = du_patch.evaluate(u, v);
            auto dv_p = dv_patch.evaluate(u, v);

            auto duv_0 = duv_patch.evaluate(u, v);
            auto duv_1 = dvu_patch.evaluate(u, v);
            auto duv_2 = duv_patch_explicit.evaluate(u, v);

            REQUIRE_THAT((du - du_p).norm(), Catch::Matchers::WithinAbs(0.0, tol));
            REQUIRE_THAT((dv - dv_p).norm(), Catch::Matchers::WithinAbs(0.0, tol));

            REQUIRE_THAT((duv_0 - duv_2).norm(), Catch::Matchers::WithinAbs(0.0, tol));
            REQUIRE_THAT((duv_1 - duv_2).norm(), Catch::Matchers::WithinAbs(0.0, tol));
        }
    }
}

/**
 * Validate 2nd derivative computation using finite difference.
 */
template <typename CurveType>
void validate_2nd_derivatives(
    const CurveType& curve, int num_samples, const typename CurveType::Scalar tol = 1e-6)
{
    using Scalar = typename CurveType::Scalar;

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> samples;
    const Scalar t_min = curve.get_domain_lower_bound();
    const Scalar t_max = curve.get_domain_upper_bound();
    samples.setLinSpaced(num_samples + 2, t_min, t_max);
    // constexpr Scalar delta = std::numeric_limits<Scalar>::epsilon() * 1e3;
    const Scalar delta = (t_max - t_min) * 1e-6;

    for (int i = 0; i < num_samples + 2; i++) {
        auto t = samples[i];
        auto d = curve.evaluate_2nd_derivative(t);

        if (i == 0) {
            auto p0 = curve.evaluate_derivative(t_min);
            auto p1 = curve.evaluate_derivative(t_min + delta);
            REQUIRE_THAT(d[0] * delta, Catch::Matchers::WithinAbs(p1[0] - p0[0], tol));
            REQUIRE_THAT(d[1] * delta, Catch::Matchers::WithinAbs(p1[1] - p0[1], tol));
        } else if (i == num_samples + 1) {
            t = t_max;
            auto p0 = curve.evaluate_derivative(t - delta);
            auto p1 = curve.evaluate_derivative(t);
            REQUIRE_THAT(d[0] * delta, Catch::Matchers::WithinAbs(p1[0] - p0[0], tol));
            REQUIRE_THAT(d[1] * delta, Catch::Matchers::WithinAbs(p1[1] - p0[1], tol));
        } else {
            // Center difference.
            const auto t0 = std::max(t_min, t - delta / 2);
            const auto t1 = std::min(t_max, t + delta / 2);
            const auto diff = t1 - t0;
            auto p0 = curve.evaluate_derivative(t0);
            auto p1 = curve.evaluate_derivative(t1);
            REQUIRE_THAT(d[0] * diff, Catch::Matchers::WithinAbs(p1[0] - p0[0], tol));
            REQUIRE_THAT(d[1] * diff, Catch::Matchers::WithinAbs(p1[1] - p0[1], tol));
        }
    }
}

/**
 * Validate 2nd derivative computation using hodograph.
 */
template <typename Scalar, int dim, int degree, bool generic>
void validate_2nd_derivatives(
    const Bezier<Scalar, dim, degree, generic>& curve, int num_samples, const Scalar tol = 1e-6)
{
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> samples;
    const Scalar t_min = curve.get_domain_lower_bound();
    const Scalar t_max = curve.get_domain_upper_bound();
    samples.setLinSpaced(num_samples + 2, t_min, t_max);

    auto hodograph = compute_hodograph(curve);
    auto hodograph2 = compute_hodograph(hodograph);

    for (int i = 0; i < num_samples + 2; i++) {
        auto t = samples[i];
        auto d = curve.evaluate_2nd_derivative(t);
        auto c = hodograph2.evaluate(t);
        REQUIRE_THAT((d - c).norm(), Catch::Matchers::WithinAbs(0.0, tol));
    }
}

/**
 * Validate 2nd derivative computation using hodograph.
 */
template <typename Scalar, int dim, int degree, bool generic>
void validate_2nd_derivatives(
    const BSpline<Scalar, dim, degree, generic>& curve, int num_samples, const Scalar tol = 1e-6)
{
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> samples;
    const Scalar t_min = curve.get_domain_lower_bound();
    const Scalar t_max = curve.get_domain_upper_bound();
    samples.setLinSpaced(num_samples + 2, t_min, t_max);

    auto hodograph = compute_hodograph(curve);
    auto hodograph2 = compute_hodograph(hodograph);

    for (int i = 0; i < num_samples + 2; i++) {
        auto t = samples[i];
        auto d = curve.evaluate_2nd_derivative(t);
        auto c = hodograph2.evaluate(t);
        REQUIRE_THAT((d - c).norm(), Catch::Matchers::WithinAbs(0.0, tol));
    }
}

template <typename CurveType, typename CurveType2>
void validate_hodograph(const CurveType& curve,
    const CurveType2& hodograph,
    int num_samples,
    const typename CurveType::Scalar tol = 1e-6)
{
    using Scalar = typename CurveType::Scalar;

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> samples;
    const Scalar t_min = curve.get_domain_lower_bound();
    const Scalar t_max = curve.get_domain_upper_bound();
    samples.setLinSpaced(num_samples + 2, t_min, t_max);

    for (int i = 0; i < num_samples + 2; i++) {
        auto t = samples[i];
        auto d = curve.evaluate_derivative(t);
        auto d2 = hodograph.evaluate(t);
        REQUIRE_THAT((d - d2).norm(), Catch::Matchers::WithinAbs(0.0, tol));
    }
}

template <typename PatchType>
void validate_iso_curves(const PatchType& patch, int num_samples = 10)
{
    const auto u_min = patch.get_u_lower_bound();
    const auto u_max = patch.get_u_upper_bound();
    const auto v_min = patch.get_v_lower_bound();
    const auto v_max = patch.get_v_upper_bound();

    for (int i = 0; i <= num_samples; i++) {
        for (int j = 0; j <= num_samples; j++) {
            const auto u = i * (u_max - u_min) / (num_samples) + u_min;
            const auto v = j * (v_max - v_min) / (num_samples) + v_min;
            auto u_curve = patch.compute_iso_curve_u(v);
            auto v_curve = patch.compute_iso_curve_v(u);
            auto p = patch.evaluate(u, v);
            auto p1 = u_curve.evaluate(u);
            auto p2 = v_curve.evaluate(v);
            REQUIRE_THAT((p - p1).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
            REQUIRE_THAT((p - p2).norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
        }
    }
}

template <typename CurveType>
void validate_approximate_inverse_evaluation(const CurveType& curve, int num_samples)
{
    const auto t_min = curve.get_domain_lower_bound();
    const auto t_max = curve.get_domain_upper_bound();
    for (int i = 0; i <= num_samples; i++) {
        const auto t = i * (t_max - t_min) / num_samples + t_min;
        const auto q = curve.evaluate(t);
        const auto t2 = curve.approximate_inverse_evaluate(q, t_min, t_max);
        const auto p = curve.evaluate(t2);
        REQUIRE_THAT((p - q).norm(), Catch::Matchers::WithinAbs(0.0, 1e-12));
    }
}

template <typename PatchType>
void validate_inverse_evaluation(const PatchType& patch,
    int u_samples,
    int v_samples,
    typename PatchType::Scalar TOL = std::numeric_limits<typename PatchType::Scalar>::epsilon() *
                                     500)
{
    const auto u_min = patch.get_u_lower_bound();
    const auto u_max = patch.get_u_upper_bound();
    const auto v_min = patch.get_v_lower_bound();
    const auto v_max = patch.get_v_upper_bound();

    for (int i = 0; i <= u_samples; i++) {
        for (int j = 0; j <= v_samples; j++) {
            const auto u = i * (u_max - u_min) / (u_samples) + u_min;
            const auto v = j * (v_max - v_min) / (v_samples) + v_min;

            const auto q = patch.evaluate(u, v);
            typename PatchType::UVPoint uv;
            bool converged;
            std::tie(uv, converged) = patch.inverse_evaluate(q, u_min, u_max, v_min, v_max);
            const auto p = patch.evaluate(uv[0], uv[1]);
            REQUIRE_THAT((p - q).norm(), Catch::Matchers::WithinAbs(0.0, TOL));
        }
    }
}

template <typename PatchType>
void validate_inverse_evaluation_3d(const PatchType& patch, int u_samples, int v_samples)
{
    const auto u_min = patch.get_u_lower_bound();
    const auto u_max = patch.get_u_upper_bound();
    const auto v_min = patch.get_v_lower_bound();
    const auto v_max = patch.get_v_upper_bound();
    validate_inverse_evaluation_3d(patch, u_samples, v_samples, {u_min, u_max, v_min, v_max});
}

template <typename PatchType>
void validate_inverse_evaluation_3d(const PatchType& patch,
    int u_samples,
    int v_samples,
    const std::array<typename PatchType::Scalar, 4>& domain)
{
    const auto u_min = domain[0];
    const auto u_max = domain[1];
    const auto v_min = domain[2];
    const auto v_max = domain[3];

    for (int i = 0; i <= u_samples; i++) {
        for (int j = 0; j <= v_samples; j++) {
            const auto u = i * (u_max - u_min) / (u_samples) + u_min;
            const auto v = j * (v_max - v_min) / (v_samples) + v_min;

            auto q = patch.evaluate(u, v);
            auto q_u = patch.evaluate_derivative_u(u, v);
            auto q_v = patch.evaluate_derivative_v(u, v);
            auto n = q_u.cross(q_v);
            if (n.norm() < 1e-12) {
                // Singular pt. Skipping for now.
                continue;
            }
            n = n / n.norm();
            q = q + .05 * n;
            typename PatchType::UVPoint uv;
            bool converged;
            std::tie(uv, converged) = patch.inverse_evaluate(q, u_min, u_max, v_min, v_max);
            const auto p = patch.evaluate(uv[0], uv[1]);
            if (std::abs((p - q).norm() - 0.05) > 1e-13) {
                nanospline::Line<typename PatchType::Scalar, 3> line;
                line.set_location(p);
                line.set_direction(q - p);
                line.initialize();
                nanospline::Line<typename PatchType::Scalar, 3> line2;
                line2.set_location(q);
                line2.set_direction(-n);
                line2.set_domain_upper_bound(0.05);
                line2.initialize();
                save_msh<typename PatchType::Scalar>("validate.msh", {&line, &line2}, {&patch});
                // assert(false);
            }
            CHECK_THAT((p - q).norm(), Catch::Matchers::WithinAbs(0.05, 1e-12));
            if (patch.get_periodic_u()) {
                const auto u_period = patch.get_u_upper_bound() - patch.get_u_lower_bound();
                const auto rounded_u = std::round((uv[0] - u) / u_period) * u_period + u;
                CHECK_THAT(std::fmod(rounded_u - uv[0], u_period), Catch::Matchers::WithinAbs(0, 1e-8));
            } else {
                CHECK_THAT(std::abs(uv[0] - u), Catch::Matchers::WithinAbs(0, 1e-8));
            }
            if (patch.get_periodic_v()) {
                const auto v_period = patch.get_v_upper_bound() - patch.get_v_lower_bound();
                const auto rounded_v = std::round((uv[1] - v) / v_period) * v_period + v;
                CHECK_THAT(std::fmod(rounded_v - uv[1], v_period), Catch::Matchers::WithinAbs(0, 1e-8));
            } else {
                CHECK_THAT(std::abs(uv[1] - v), Catch::Matchers::WithinAbs(0, 1e-8));
            }
        }
    }
}

template <typename Scalar, int dim>
void offset_and_validate(const CurveBase<Scalar, dim>& curve1, int num_samples = 10)
{
    constexpr Scalar tol = static_cast<Scalar>(1e-6);
    using Point = Eigen::Matrix<Scalar, 1, dim>;

    Point offset;
    offset.setOnes();
    const Scalar t_offset = (curve1.get_num_knots() == 0) ? 0 : 1;

    auto curve2_ptr = curve1.clone();
    auto& curve2 = *curve2_ptr;

    REQUIRE(curve2.get_num_control_points() == curve1.get_num_control_points());
    REQUIRE(curve2.get_num_weights() == curve1.get_num_weights());

    for (int i = 0; i < curve2.get_num_control_points(); i++) {
        auto p = curve2.get_control_point(i);
        curve2.set_control_point(i, p + offset);
    }

    for (int i = 0; i < curve2.get_num_weights(); i++) {
        auto w = curve2.get_weight(i);
        curve2.set_weight(i, w * 2);
    }

    for (int i = 0; i < curve2.get_num_knots(); i++) {
        auto t = curve2.get_knot(i);
        curve2.set_knot(i, t + t_offset);
    }

    const Scalar t_min = curve1.get_domain_lower_bound();
    const Scalar t_max = curve1.get_domain_upper_bound();

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> samples;
    samples.setLinSpaced(num_samples + 2, t_min, t_max);

    for (int i = 0; i < num_samples + 2; i++) {
        const auto p1 = curve1.evaluate(samples[i]);
        const auto p2 = curve2.evaluate(samples[i] + t_offset);
        REQUIRE_THAT((p1 - p2).norm(), Catch::Matchers::WithinAbs(offset.norm(), tol));
    }
}

template <typename Scalar, int dim>
void offset_and_validate(const PatchBase<Scalar, dim>& patch1, int num_samples = 10)
{
    constexpr Scalar tol = static_cast<Scalar>(1e-6);
    using Point = Eigen::Matrix<Scalar, 1, dim>;

    Point offset;
    offset.setOnes();
    const Scalar u_offset = (patch1.get_num_knots_u() == 0) ? 0 : 1;
    const Scalar v_offset = (patch1.get_num_knots_v() == 0) ? 0 : 2;

    auto patch2_ptr = patch1.clone();
    auto& patch2 = *patch2_ptr;

    REQUIRE(patch2.num_control_points() == patch1.num_control_points());
    REQUIRE(patch2.get_num_weights_u() == patch1.get_num_weights_u());
    REQUIRE(patch2.get_num_weights_v() == patch1.get_num_weights_v());

    for (int i = 0; i < patch2.num_control_points_u(); i++) {
        for (int j = 0; j < patch2.num_control_points_v(); j++) {
            auto p = patch2.get_control_point(i, j);
            patch2.set_control_point(i, j, p + offset);
        }
    }

    for (int i = 0; i < patch2.get_num_weights_u(); i++) {
        for (int j = 0; j < patch2.get_num_weights_v(); j++) {
            auto w = patch2.get_weight(i, j);
            patch2.set_weight(i, j, w * 2);
        }
    }

    for (int i = 0; i < patch2.get_num_knots_u(); i++) {
        auto u = patch2.get_knot_u(i);
        patch2.set_knot_u(i, u + u_offset);
    }

    for (int i = 0; i < patch2.get_num_knots_v(); i++) {
        auto v = patch2.get_knot_v(i);
        patch2.set_knot_v(i, v + v_offset);
    }

    const Scalar u_min = patch2.get_u_lower_bound();
    const Scalar u_max = patch2.get_u_upper_bound();
    const Scalar v_min = patch2.get_v_lower_bound();
    const Scalar v_max = patch2.get_v_upper_bound();

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> u_samples;
    u_samples.setLinSpaced(num_samples + 2, u_min, u_max);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> v_samples;
    v_samples.setLinSpaced(num_samples + 2, v_min, v_max);

    for (int i = 0; i < num_samples + 2; i++) {
        for (int j = 0; j < num_samples + 2; j++) {
            const auto p1 = patch1.evaluate(u_samples[i], v_samples[j]);
            const auto p2 = patch2.evaluate(u_samples[i] + u_offset, v_samples[j] + v_offset);
            REQUIRE_THAT((p1 - p2).norm(), Catch::Matchers::WithinAbs(offset.norm(), tol));
        }
    }
}


} // namespace nanospline
