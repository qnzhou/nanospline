#pragma once

#include <nanospline/CurveBase.h>
#include <nanospline/PatchBase.h>

#include <array>
#include <limits>

namespace nanospline {

/**
 * Compute curve-plane intersection using Newton's method.
 *
 * @param[in]  curve  Input 3D curve.
 * @param[in]  plane  Coefficients [a,b,c,d] of a 3D plane equation: ax+by+cz+d=0.
 * @param[in]  t0     Initial guess of the intersection on the input curve.
 * @param[in]  num_iterations  The max number of iterations to use.
 * @param[in]  tol    The distance tolerance for determining convergence.
 *
 * @return [t, converged],where the parametric value t representing the curve-plane intersection,
 * and converged indicates whether the Newton iterations have converged.
 */
template <typename Scalar>
auto intersect(const CurveBase<Scalar, 3>& curve,
    const std::array<Scalar, 4>& plane,
    Scalar t0,
    int num_iterations,
    Scalar tol) -> std::tuple<Scalar, bool>
{
    Scalar prev_t = t0;
    Scalar t = t0;
    Scalar prev_err = std::numeric_limits<Scalar>::max();
    for (int i = 0; i < num_iterations; i++) {
        const auto d0 = curve.evaluate(t);
        Scalar err = plane[0] * d0[0] + plane[1] * d0[1] + plane[2] * d0[2] + plane[3];
        if (std::abs(err) < tol) return {t, true};
        if (std::abs(err) > prev_err) {
            return {prev_t, false};
        }

        prev_err = std::abs(err);
        prev_t = t;

        const auto d1 = curve.evaluate_derivative(t);
        Scalar err_derivative = plane[0] * d1[0] + plane[1] * d1[1] + plane[2] * d1[2];
        t = t - err / err_derivative;

        if (!std::isfinite(t)) {
            // In case err_derivative is near zero and cause numerical problems.
            return {prev_t, false};
        }
    }

    return {t, false};
}

/**
 * Compute implicit curve-plane intersection using Newton's method.  The input
 * curve is defined as the image of a line segment in UV space on a patch.
 *
 * @param[in]  patch  Input 3D patch.
 * @param[in]  pu,pv  Starting UV point of the curve.
 * @param[in]  qu,qv  Ending UV point of the curve.
 * @param[in]  plane  Coefficients [a,b,c,d] of a 3D plane equation: ax+by+cz+d=0.
 * @param[in]  u0,v0  Initial guess of the intersection on the input curve.
 * @param[in]  num_iterations  The max number of iterations to use.
 * @param[in]  tol    The distance tolerance for determining convergence.
 *
 * @return [u,v,converged] where [u,v] is the parametric values representing the
 * curve-plane intersection, and `converged` indicate whether the Newton
 * iterations have converged.
 */
template <typename Scalar>
auto intersect(const PatchBase<Scalar, 3>& patch,
    Scalar pu,
    Scalar pv,
    Scalar qu,
    Scalar qv,
    const std::array<Scalar, 4>& plane,
    Scalar u0,
    Scalar v0,
    int num_iterations,
    Scalar tol) -> std::tuple<Scalar, Scalar, bool>
{
    using Point = typename PatchBase<Scalar, 3>::Point;

    auto interpolate = [&](Scalar t) -> std::tuple<Scalar, Scalar> {
        return {pu * (1 - t) + qu * t, pv * (1 - t) + qv * t};
    };

    if (std::abs(pu - qu) < 1e-12 && std::abs(pv - qv) < 1e-12) {
        // Curve degenerates to a point.  Simply check if it is on the line.
        Point p = patch.evaluate(pu, pv);
        Scalar err = plane[0] * p[0] + plane[1] * p[1] + plane[2] * p[2] + plane[3];
        if (err < tol)
            return {pu, pv, true};
        else
            return {pu, pv, false};
    }

    Scalar prev_t = 1 - std::hypot(u0 - pu, v0 - pv) / std::hypot(pu - qu, pv - qv);
    Scalar t = prev_t;
    Scalar u = u0, v = v0;
    Scalar prev_err = std::numeric_limits<Scalar>::max();
    Point d0, d1, du, dv;

    for (int i = 0; i < num_iterations; i++) {
        std::tie(u, v) = interpolate(t);

        d0 = patch.evaluate(u, v);
        Scalar err = plane[0] * d0[0] + plane[1] * d0[1] + plane[2] * d0[2] + plane[3];
        if (std::abs(err) < tol) return {u, v, true};
        if (std::abs(err) > prev_err) {
            std::tie(u, v) = interpolate(prev_t);
            return {u, v, false};
        }

        prev_err = std::abs(err);
        prev_t = t;

        du = patch.evaluate_derivative_u(u, v);
        dv = patch.evaluate_derivative_v(u, v);
        d1 = du * (qu - pu) + dv * (qv - pv);

        Scalar err_derivative = plane[0] * d1[0] + plane[1] * d1[1] + plane[2] * d1[2];
        t = t - err / err_derivative;
        if (!std::isfinite(t)) {
            // In case err_derivative is near zero and cause numerical problems.
            std::tie(u, v) = interpolate(prev_t);
            return {u, v, false};
        }
    }

    return {u, v, false};
}

/**
 * Compute intersection of a generic curve and a generic patch.
 *
 * @param[in]  curve  Input curve.
 * @param[in]  patch  Input patch.
 * @param[in]  t0     Initial guess of intersection on curve.
 * @param[in]  u0,v0  Initial guess of intersection on patch.
 * @param[in]  num_interation  The max number of iterations to use.
 * @param[in]  tol    The distance tolerance for determining convergence.
 *
 * @return A tuple [t, u, v, converged], where t, and (u,v) represents the
 * intersection on curve and patch respectively, and `converged` indicates
 * whether the Newton iterations have converged.
 */
template <typename Scalar>
auto intersect(const CurveBase<Scalar, 3>& curve,
    const PatchBase<Scalar, 3>& patch,
    Scalar t0,
    Scalar u0,
    Scalar v0,
    int num_iterations,
    Scalar tol) -> std::tuple<Scalar, Scalar, Scalar, bool>
{
    using Vector = Eigen::Matrix<Scalar, 3, 1>;
    using Matrix = Eigen::Matrix<Scalar, 3, 3>;

    auto evaluate_objective_and_derivatives =
        [&](Scalar t, Scalar u, Scalar v) -> std::tuple<Scalar, Vector, Matrix> {
        auto c = curve.evaluate(t);
        auto p = patch.evaluate(u, v);
        auto l = c - p;

        auto dt = curve.evaluate_derivative(t);
        auto du = patch.evaluate_derivative_u(u, v);
        auto dv = patch.evaluate_derivative_v(u, v);

        auto dtt = curve.evaluate_2nd_derivative(t);
        auto duu = patch.evaluate_2nd_derivative_uu(u, v);
        auto duv = patch.evaluate_2nd_derivative_uv(u, v);
        auto dvv = patch.evaluate_2nd_derivative_vv(u, v);

        Matrix H;
        H(0, 0) = dtt.dot(l) + dt.squaredNorm();
        H(0, 1) = -dt.dot(du);
        H(0, 2) = -dt.dot(dv);
        H(1, 0) = H(0, 1);
        H(1, 1) = duu.dot(-l) + du.squaredNorm();
        H(1, 2) = duv.dot(-l) + du.dot(dv);
        H(2, 0) = H(0, 2);
        H(2, 1) = H(1, 2);
        H(2, 2) = dvv.dot(-l) + dv.squaredNorm();

        Scalar f = l.squaredNorm();
        Vector g(dt.dot(l), -du.dot(l), -dv.dot(l));
        return {f, g, H};
    };

    Scalar t = t0, prev_t = t0;
    Scalar u = u0, prev_u = u0;
    Scalar v = v0, prev_v = v0;
    Scalar f, prev_f = std::numeric_limits<Scalar>::max();
    Vector g;
    Matrix H;
    for (int i = 0; i < num_iterations; i++) {
        std::tie(f, g, H) = evaluate_objective_and_derivatives(t, u, v);
        if (f > prev_f) {
            return {prev_t, prev_u, prev_v, false};
        }
        if (f < tol * tol) {
            return {t, u, v, true};
        }

        prev_t = t;
        prev_u = u;
        prev_v = v;
        prev_f = f;

        Vector delta = H.inverse() * (-g);
        if (!delta.array().isFinite().all()) {
            return {prev_t, prev_u, prev_v, false};
        }
        t += delta[0];
        u += delta[1];
        v += delta[2];
    }

    return {t, u, v, false};
}

} // namespace nanospline
