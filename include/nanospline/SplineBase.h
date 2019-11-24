#pragma once

#include <cassert>
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

        virtual Scalar approximate_inverse_evaluate(const Point& p,
                const Scalar lower=0.0,
                const Scalar upper=1.0,
                const int level=3) const =0;

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
