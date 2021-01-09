#pragma once

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include <nanospline/Exceptions.h>
#include <Eigen/Core>
namespace nanospline {

template <typename _Scalar, int _dim>
class CurveBase
{
public:
    static_assert(_dim >= 0, "Negative degree is not allowed");
    using Scalar = _Scalar;
    using Point = Eigen::Matrix<Scalar, 1, _dim>;

public:
    virtual ~CurveBase() = default;
    virtual std::unique_ptr<CurveBase<_Scalar, _dim>> clone() const = 0;

    virtual void initialize() {}

    virtual void initialize() {}

    constexpr int get_dim() const { return _dim; }
    virtual int get_degree() const = 0;
    virtual bool in_domain(Scalar t) const = 0;
    virtual Scalar get_domain_lower_bound() const = 0;
    virtual Scalar get_domain_upper_bound() const = 0;

    void set_periodic(bool periodic) { m_periodic = periodic; }

    bool get_periodic() const { return m_periodic; }

    bool is_closed(int order = 0, const Scalar tol = std::numeric_limits<Scalar>::epsilon()) const
    {
        const Scalar t_min = get_domain_lower_bound();
        const Scalar t_max = get_domain_upper_bound();

        const Scalar d0 = (evaluate(t_min) - evaluate(t_max)).norm();
        bool result = d0 < tol;
        if (order >= 1) {
            const Scalar d1 = (evaluate_derivative(t_min) - evaluate_derivative(t_max)).norm();
            result = result && d1 < tol;
        }
        if (order >= 2) {
            const Scalar d2 =
                (evaluate_2nd_derivative(t_min) - evaluate_2nd_derivative(t_max)).norm();
            result = result && d2 < tol;
        }
        return result;
    }

public:
    virtual Point evaluate(Scalar t) const = 0;
    virtual Scalar inverse_evaluate(const Point& p) const = 0;
    virtual Point evaluate_derivative(Scalar t) const = 0;
    virtual Point evaluate_2nd_derivative(Scalar t) const = 0;
    virtual Scalar approximate_inverse_evaluate(const Point& p,
        const Scalar lower = 0.0,
        const Scalar upper = 1.0,
        const int level = 3) const = 0;

    Point evaluate_curvature(Scalar t) const
    {
        const auto d1 = evaluate_derivative(t);
        const auto d2 = evaluate_2nd_derivative(t);

        const auto sq_speed = d1.squaredNorm();
        if (sq_speed == 0) {
            return Point::Zero();
        } else {
            return (d2 - d1 * (d1.dot(d2)) / sq_speed) / sq_speed;
        }
    }

public:
    virtual int get_num_control_points() const = 0;
    virtual Point get_control_point(int i) const = 0;
    virtual void set_control_point(int i, const Point& p) = 0;

    virtual int get_num_weights() const = 0;
    virtual Scalar get_weight(int i) const = 0;
    virtual void set_weight(int i, Scalar val) = 0;

    virtual int get_num_knots() const = 0;
    virtual Scalar get_knot(int i) const = 0;
    virtual void set_knot(int i, Scalar val) = 0;

public:
    virtual std::vector<Scalar> compute_inflections(
        const Scalar lower = 0.0, const Scalar upper = 1.0) const = 0;

    virtual std::vector<Scalar> reduce_turning_angle(
        const Scalar lower = 0.0, const Scalar upper = 1.0) const = 0;

    virtual std::vector<Scalar> compute_singularities(
        const Scalar lower = 0.0, const Scalar upper = 1.0) const = 0;

    virtual Scalar get_turning_angle(Scalar t0, Scalar t1) const
    {
        if (_dim != 2) {
            throw std::runtime_error("Turning angle computation is for 2D curves only");
        }

        constexpr Scalar EPS = std::numeric_limits<Scalar>::epsilon();
        constexpr int NUM_RETRIES = 10;
        Point d0 = evaluate_derivative(t0);
        Point d1 = evaluate_derivative(t1);

        for (int i = 0; i < NUM_RETRIES && d0.norm() < EPS; i++) {
            t0 += (t1 - t0) * 1e-3;
            d0 = evaluate_derivative(t0);
        }

        for (int i = 0; i < NUM_RETRIES && d1.norm() < EPS; i++) {
            t1 -= (t1 - t0) * 1e-3;
            d1 = evaluate_derivative(t1);
        }

        return std::atan2(d0[0] * d1[1] - d0[1] * d1[0], d0[0] * d1[0] + d0[1] * d1[1]);
    }

protected:
    bool is_split_point_valid(Scalar t) const
    {
        if (t <= get_domain_lower_bound() || t >= get_domain_upper_bound()) {
            return false;
        } else {
            return true;
        }
    }

    Scalar approximate_inverse_evaluate(const Point& p,
        const int num_samples,
        const Scalar lower = 0.0,
        const Scalar upper = 1.0,
        const int level = 3) const
    {
        assert(num_samples > 0);
        const Scalar delta_t = (upper - lower) / num_samples;

        Scalar min_t = 0.0;
        Scalar min_dist = std::numeric_limits<Scalar>::max();
        for (int i = 0; i <= num_samples; i++) {
            const Scalar t = lower + (Scalar)(i) / (Scalar)(num_samples) * (upper - lower);
            const auto q = this->evaluate(t);
            const auto dist = (p - q).squaredNorm();
            if (dist < min_dist) {
                min_dist = dist;
                min_t = t;
            }
        }

        if (level <= 0) {
            constexpr Scalar TOL = std::numeric_limits<Scalar>::epsilon() * 100;
            return newton_raphson(p, min_t, 10, TOL, lower, upper);
        } else {
            return approximate_inverse_evaluate(p,
                num_samples,
                std::max(min_t - delta_t, lower),
                std::min(min_t + delta_t, upper),
                level - 1);
        }
    }

    Scalar newton_raphson(const Point& p,
        Scalar t,
        int num_iterations,
        const Scalar tol,
        const Scalar lower,
        const Scalar upper) const
    {
        Scalar prev_t = t;
        Scalar prev_err = -1;
        for (int i = 0; i < num_iterations; i++) {
            const auto d0 = this->evaluate(t);
            const auto d1 = this->evaluate_derivative(t);
            const auto d2 = this->evaluate_2nd_derivative(t);
            const auto f = (p - d0).dot(d1);
            const auto df = (p - d0).dot(d2) - d1.squaredNorm();
            const auto err = std::abs(f);
            if (err < tol) return t;
            if (prev_err > 0 && err > prev_err) return prev_t;

            prev_err = err;
            prev_t = t;

            t -= f / df;
            if (t <= lower) return lower;
            if (t >= upper) return upper;
        }
        return t;
    }

    Scalar unwrap_parameter(Scalar t) const
    {
        assert(m_periodic);
        const Scalar t_min = get_domain_lower_bound();
        const Scalar t_max = get_domain_upper_bound();
        const Scalar period = t_max - t_min;
        t = fmod(t - t_min, period) + t_min;
        if (t < t_min) t += period;
        assert(t >= t_min && t <= t_max);
        return t;
    }

private:
    bool m_periodic = false;
};

} // namespace nanospline
