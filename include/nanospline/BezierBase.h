#pragma once

#include <Eigen/Core>

#include <nanospline/SplineBase.h>

namespace nanospline {

template<typename _Scalar, int _dim, int _degree, bool _generic>
class BezierBase : public SplineBase<_Scalar, _dim> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        static_assert(_degree>=0 || _generic,
                "Invalid degree for non-generic Bezier setting");
        using Base = SplineBase<_Scalar, _dim>;
        using Scalar = _Scalar;
        using Point = Eigen::Matrix<Scalar, 1, _dim>;
        using ControlPoints = Eigen::Matrix<Scalar, _generic?Eigen::Dynamic:_degree+1, _dim>;

    public:
        virtual ~BezierBase()=default;
        virtual Point evaluate(Scalar t) const =0;
        virtual Scalar inverse_evaluate(const Point& p) const =0;

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

    protected:
        ControlPoints m_control_points;
};

}
