#pragma once

#include <Eigen/Core>

namespace nanospline {

template<typename DerivedSpline>
class SplineBase {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        using Scalar = typename DerivedSpline::Scalar;
        using Point = typename DerivedSpline::Point;
        using ControlPoints = typename DerivedSpline::ControlPoints;

    public:
        virtual ~SplineBase() = default;
        virtual Point evaluate(Scalar t) const =0;
        virtual Scalar inverse_evaluate(const Point& p) const =0;

    public:
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
