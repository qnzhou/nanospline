#pragma once

#include <Eigen/Core>

#include <nanospline/SplineBase.h>

namespace nanospline {

template<typename _Scalar, int _dim, int _degree, bool _generic>
class BSplineBase : public SplineBase<_Scalar, _dim> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        static_assert(_degree>=0 || _generic,
                "Invalid degree for non-generic B-spline setting");
        using Base = SplineBase<_Scalar, _dim>;
        using Scalar = _Scalar;
        using Point = Eigen::Matrix<Scalar, 1, _dim>;
        using ControlPoints = Eigen::Matrix<Scalar, Eigen::Dynamic, _dim>;
        using KnotVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    public:
        virtual ~BSplineBase()=default;
        virtual Point evaluate(Scalar t) const=0;
        virtual Scalar inverse_evaluate(const Point& p) const=0;

    public:
        Scalar approximate_inverse_evaluate(const Point& p,
                const Scalar lower=0.0,
                const Scalar upper=1.0,
                const int level=3) const {
            const int num_samples = 2 *
                (_generic ? m_control_points.rows() : _degree+1);

            return Base::approximate_inverse_evaluate(
                    p, num_samples, lower, upper, level);
        }

        int locate_span(const Scalar t) const {
            assert(m_knots.rows() > m_control_points.rows());
            int low = m_knots.rows() - m_control_points.rows() - 1;
            int high = m_control_points.rows();
            assert(m_knots[low] <= t);
            assert(m_knots[high] >= t);

            if (t == m_knots[high]) return high-1;

            int mid = (high+low) / 2;
            while(t < m_knots[mid] || t >= m_knots[mid+1]) {
                if (t < m_knots[mid]) high=mid;
                else low = mid;
                mid = (high+low) / 2;
            }

            return mid;
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
