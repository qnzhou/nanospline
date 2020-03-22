#pragma once

#include <cassert>
#include <cmath>
#include <limits>
#include <vector>
#include <memory>

#include <Eigen/Core>

namespace nanospline {

template<typename _Scalar, int _dim>
class CurveBase {
    public:
        static_assert(_dim >= 0, "Negative degree is not allowed");
        using Scalar = _Scalar;
        using Point = Eigen::Matrix<Scalar, 1, _dim>;

    public:
        virtual ~CurveBase()=default;
        virtual Point evaluate(Scalar t) const =0;
        virtual Scalar inverse_evaluate(const Point& p) const =0;
        virtual Point evaluate_derivative(Scalar t) const =0;
        virtual Point evaluate_2nd_derivative(Scalar t) const =0;

        virtual Scalar approximate_inverse_evaluate(const Point& p,
                const Scalar lower=0.0,
                const Scalar upper=1.0,
                const int level=3) const =0;

        virtual std::vector<Scalar> compute_inflections(
                const Scalar lower=0.0,
                const Scalar upper=1.0) const =0;

        virtual std::vector<Scalar> reduce_turning_angle(
                const Scalar lower=0.0,
                const Scalar upper=1.0) const =0;

        virtual void write(std::ostream &out) const =0;

        friend std::ostream &operator<<(std::ostream &out, const CurveBase &c) { c.wirte(out); return out; }
        virtual std::shared_ptr<CurveBase> simplify(Scalar eps) const { return nullptr; }
        // virtual bool is_point() const = 0;

        virtual Scalar get_turning_angle(Scalar t0, Scalar t1) const {
            if (_dim != 2) {
                throw std::runtime_error(
                        "Inflection computation is for 2D curves only");
            }

            constexpr Scalar EPS = std::numeric_limits<Scalar>::epsilon();
            Scalar turning_angle = 0;
            constexpr auto num_samples = 10;
            Point prev_d(0, 0);

            for (int i=0; i<num_samples; i++) {
                const Scalar t = t0 + (t1-t0) * i / (num_samples-1);
                const auto d = evaluate_derivative(t);
                if (prev_d.norm() < EPS) {
                    prev_d = d;
                    continue;
                }

                if (d.norm() < EPS) {
                    continue;
                }

                turning_angle += std::atan2(
                        d[0]*prev_d[1] - d[1]*prev_d[0],
                        d[0]*prev_d[0] + d[1]*prev_d[1]);
                prev_d = d;
            }
            return turning_angle;
        }

    public:
        Point evaluate_curvature(Scalar t) const {
            const auto d1 = evaluate_derivative(t);
            const auto d2 = evaluate_2nd_derivative(t);

            const auto sq_speed = d1.squaredNorm();
            if (sq_speed == 0) {
                return Point::Zero();
            } else {
                return (d2 - d1 * (d1.dot(d2)) / sq_speed) / sq_speed;
            }
        }

    protected:
        Scalar approximate_inverse_evaluate(const Point& p,
                const int num_samples,
                const Scalar lower=0.0,
                const Scalar upper=1.0,
                const int level=3) const {

            assert(num_samples > 0);
            const Scalar delta_t = (upper - lower) / num_samples;

            Scalar min_t = 0.0;
            Scalar min_dist = std::numeric_limits<Scalar>::max();
            for (int i=0; i<=num_samples; i++) {
                const Scalar t = lower +
                    (Scalar)(i) / (Scalar)(num_samples) * (upper-lower);
                const auto q = this->evaluate(t);
                const auto dist = (p-q).squaredNorm();
                if (dist < min_dist) {
                    min_dist = dist;
                    min_t = t;
                }
            }

            if (level <= 0) {
                return min_t;
            } else {
                return approximate_inverse_evaluate(p,
                        num_samples,
                        std::max(min_t-delta_t, lower),
                        std::min(min_t+delta_t, upper),
                        level-1);
            }
        }

};

}
