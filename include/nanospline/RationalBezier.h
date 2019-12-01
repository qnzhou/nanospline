#pragma once

#include <Eigen/Core>

#include <nanospline/Exceptions.h>
#include <nanospline/BezierBase.h>
#include <nanospline/Bezier.h>

namespace nanospline {

template<typename _Scalar, int _dim, int _degree, bool _generic>
class RationalBezier : public BezierBase<_Scalar, _dim, _degree, _generic> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        using Base = BezierBase<_Scalar, _dim, _degree, _generic>;
        using Scalar = typename Base::Scalar;
        using Point = typename Base::Point;
        using ControlPoints = typename Base::ControlPoints;
        using WeightVector = Eigen::Matrix<_Scalar, _generic?Eigen::Dynamic:_degree+1, 1>;
        using BezierHomogeneous = Bezier<_Scalar, _dim+1, _degree, _generic>;

    public:
        Point evaluate(Scalar t) const override {
            auto p = m_bezier_homogeneous.evaluate(t);
            return p.template head<_dim>() / p[_dim];
        }

        Scalar inverse_evaluate(const Point& p) const override {
            throw not_implemented_error("Too complex, sigh");
        }

        Point evaluate_derivative(Scalar t) const override {
            const auto p = m_bezier_homogeneous.evaluate(t);
            const auto d =
                m_bezier_homogeneous.evaluate_derivative(t);

            return (d.template head<_dim>() -
                    p.template head<_dim>() * d[_dim] / p[_dim])
                / p[_dim];
        }

        Point evaluate_2nd_derivative(Scalar t) const override {
            throw not_implemented_error("Too complex, sigh");
        }

    public:
        void initialize() {
            typename BezierHomogeneous::ControlPoints ctrl_pts(
                    Base::m_control_points.rows(), _dim+1);
            ctrl_pts.template leftCols<_dim>() =
                Base::m_control_points.array().colwise() * m_weights.array();
            ctrl_pts.template rightCols<1>() = m_weights;

            m_bezier_homogeneous.set_control_points(std::move(ctrl_pts));
        }

        const WeightVector& get_weights() const {
            return m_weights;
        }

        template<typename Derived>
        void set_weights(const Eigen::PlainObjectBase<Derived>& weights) {
            m_weights = weights;
        }

        template<typename Derived>
        void set_weights(const Eigen::PlainObjectBase<Derived>&& weights) {
            m_weights.swap(weights);
        }

        const BezierHomogeneous& get_homogeneous() const {
            return m_bezier_homogeneous;
        }

        void set_homogeneous(const BezierHomogeneous& homogeneous) {
            const auto ctrl_pts = homogeneous.get_control_points();
            m_bezier_homogeneous = homogeneous;
            m_weights = ctrl_pts.template rightCols<1>();
            Base::m_control_points =
                ctrl_pts.template leftCols<_dim>().array().colwise()
                / m_weights.array();
        }

    private:
        BezierHomogeneous m_bezier_homogeneous;
        WeightVector m_weights;
};

}
