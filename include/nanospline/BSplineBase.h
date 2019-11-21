#pragma once

#include <Eigen/Core>
#include <iostream>

#include <nanospline/SplineBase.h>

namespace nanospline {

template<typename _Scalar, int _dim, int _degree, bool _generic>
class BSplineBase : public SplineBase<_Scalar, _dim> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        static_assert(_dim > 0, "Dimension must be positive.");
        static_assert(_degree>=0 || _generic,
                "Invalid degree for non-generic B-spline setting");
        using Base = SplineBase<_Scalar, _dim>;
        using Scalar = _Scalar;
        using Point = Eigen::Matrix<Scalar, 1, _dim>;
        using ControlPoints = Eigen::Matrix<Scalar, Eigen::Dynamic, _dim>;
        using KnotVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    public:
        virtual ~BSplineBase()=default;
        virtual Point evaluate(Scalar t) const override =0;
        virtual Scalar inverse_evaluate(const Point& p) const override =0;

        virtual Scalar approximate_inverse_evaluate(const Point& p,
                const Scalar lower=0.0,
                const Scalar upper=1.0,
                const int level=3) const override {
            assert(lower < upper);
            const int num_samples = 2 * (get_degree() + 1);

            auto curr_span = locate_span(lower);

            Scalar min_t = 0.0, min_delta = 0.0;
            Scalar min_dist = std::numeric_limits<Scalar>::max();

            Scalar curr_lower = lower;
            while (curr_lower < upper) {
                Scalar curr_upper = std::min(m_knots[curr_span+1], upper);

                if (curr_upper > curr_lower) {
                    const Scalar delta = (curr_upper - curr_lower) / num_samples;
                    for (int j=0; j<num_samples+1; j++) {
                        const Scalar r = (Scalar)(j) / (Scalar)(num_samples);
                        const Scalar t = curr_lower + r * (curr_upper - curr_lower);
                        const auto q = this->evaluate(t);
                        const auto dist = (p-q).squaredNorm();
                        if (dist < min_dist) {
                            min_dist = dist;
                            min_t = t;
                            min_delta = delta;
                        }
                    }
                }

                curr_lower = curr_upper;
                curr_span++;
            }

            if (level <= 0) {
                return min_t;
            } else {
                return approximate_inverse_evaluate(
                        p,
                        std::max(lower, min_t-min_delta),
                        std::min(upper, min_t+min_delta),
                        level-1);
            }
        }

        int locate_span(const Scalar t) const {
            const auto num_knots = m_knots.rows();
            assert(num_knots > m_control_points.rows());
            int low = 0;//m_knots.rows() - m_control_points.rows() - 1;
            int high = m_knots.rows()-1; //m_control_points.rows();
            assert(m_knots[low] <= t);
            assert(m_knots[high] >= t);

            auto bypass_duplicates_after = [this, num_knots](int i) {
                while(i+1<num_knots && this->m_knots[i] == this->m_knots[i+1]) {
                    i=i+1;
                }
                return i;
            };

            auto bypass_duplicates_before= [this](int i) {
                while(i-1>=0 && this->m_knots[i] == this->m_knots[i-1]) {
                    i=i-1;
                }
                return i;
            };

            if (t == m_knots[high]) return bypass_duplicates_before(high)-1;

            int mid = (high+low) / 2;
            while(t < m_knots[mid] || t >= m_knots[mid+1]) {
                if (t < m_knots[mid]) high=mid;
                else low = mid;
                mid = (high+low) / 2;
            }

            return bypass_duplicates_after(mid);
        }

    public:
        const ControlPoints& get_control_points() const {
            return m_control_points;
        }

        template<typename Derived>
        void set_control_points(const Eigen::PlainObjectBase<Derived>& ctrl_pts) {
            m_control_points = ctrl_pts;
        }

        template<typename Derived>
        void set_control_points(Eigen::PlainObjectBase<Derived>&& ctrl_pts) {
            m_control_points.swap(ctrl_pts);
        }

        const KnotVector get_knots() const {
            return m_knots;
        }

        template<typename Derived>
        void set_knots(const Eigen::PlainObjectBase<Derived>& knots) {
            m_knots = knots;
        }

        template<typename Derived>
        void set_knots(Eigen::PlainObjectBase<Derived>&& knots) {
            m_knots.swap(knots);
        }

        int get_degree() const {
            int degree = m_knots.rows() - m_control_points.rows() - 1;
            if (_degree >= 0) {
                assert(degree == _degree);
            }
            return degree;
        }

    protected:
        ControlPoints m_control_points;
        KnotVector m_knots;
};

}
