#pragma once

#include <cassert>
#include <cmath>
#include <limits>

#include <Eigen/Core>

namespace nanospline {

template<typename _Scalar, int _dim>
class SplineBase {
    public:
        using Scalar = _Scalar;
        using Point = Eigen::Matrix<Scalar, 1, _dim>;

    public:
        virtual ~SplineBase()=default;
        virtual Point evaluate(Scalar t) const =0;
        virtual Scalar inverse_evaluate(const Point& p) const =0;
        virtual Point evaluate_derivative(Scalar t) const =0;
        virtual Point evaluate_2nd_derivative(Scalar t) const =0;

        virtual Scalar approximate_inverse_evaluate(const Point& p,
                const Scalar lower=0.0,
                const Scalar upper=1.0,
                const int level=3) const =0;

        virtual void write(std::ostream &out) const =0;

        friend std::ostream &operator<<(std::ostream &out, const SplineBase &c) { c.wirte(out); return out; }
        virtual std::shared_ptr<SplineBase> simplify(Scalar eps) const { return nullptr; }
        // virtual std::string to_eps() const = 0;
        // virtual Scalar optimal_point_to_reduce_turning_angle(Scalar t0, Scalar t1, bool to_flip = false) const = 0;
        // virtual bool is_point() const = 0;

        Scalar get_turning_angle(Scalar t0, Scalar t1) const
        {
            using std::acos;

            Point n0 = evaluate_derivative(t0);
            n0.normalize();
            Point n1 = eval_first_derivative(t1);
            n1.normalize();

            Scalar cos_a = n0.dot(n1);
            if (cos_a > 1)
                cos_a = 1;
            if (cos_a < -1)
                cos_a = -1;

            const Scalar angle = acos(cos_a);

            return angle;
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
