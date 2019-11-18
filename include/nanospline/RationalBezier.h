#pragma once

#include <Eigen/Core>

#include <nanospline/Exceptions.h>
#include <nanospline/BezierBase.h>
#include <nanospline/Bezier.h>

namespace nanospline {

template<typename _Scalar, int _dim, int _degree, bool _generic>
class RationalBezier : public BezierBase<_Scalar, _dim, _degree, _generic> {
    public:
        using Base = BezierBase<_Scalar, _dim, _degree, _generic>;
        using Scalar = typename Base::Scalar;
        using Point = typename Base::Point;
        using ControlPoints = typename Base::ControlPoints;
        using WeightVector = Eigen::Matrix<_Scalar, _generic?Eigen::Dynamic:_degree+1, 1>;
        using BezierHomogeneous = Bezier<_Scalar, _dim+1, _degree, _generic>;

    public:
        Point evaluate(Scalar t) const override {
            auto p = m_bezier_homogeneous.evaluate(t);
            return p.template segment<_dim>(0) / p[_dim];
        }

        Scalar inverse_evaluate(const Point& p) const override {
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

    private:
        BezierHomogeneous m_bezier_homogeneous;
        WeightVector m_weights;
};

}
