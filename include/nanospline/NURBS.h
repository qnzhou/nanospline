#pragma once

#include <Eigen/Core>

#include <nanospline/Exceptions.h>
#include <nanospline/BSplineBase.h>
#include <nanospline/BSpline.h>

namespace nanospline {

template<typename _Scalar, int _dim, int _degree, bool _generic>
class NURBS : public BSplineBase<_Scalar, _dim, _degree, _generic> {
    public:
        using Base = BSplineBase<_Scalar, _dim, _degree, _generic>;
        using Scalar = typename Base::Scalar;
        using Point = typename Base::Point;
        using ControlPoints = typename Base::ControlPoints;
        using WeightVector = Eigen::Matrix<_Scalar, _generic?Eigen::Dynamic:_degree+1, 1>;
        using BSplineHomogeneous = BSpline<_Scalar, _dim+1, _degree, _generic>;

    public:
        Point evaluate(Scalar t) const override {
            auto p = m_bspline_homogeneous.evaluate(t);
            return p.template segment<_dim>(0) / p[_dim];
        }

        Scalar inverse_evaluate(const Point& p) const override {
            throw not_implemented_error("Too complex, sigh");
        }

        Point evaluate_derivative(Scalar t) const override {
            throw not_implemented_error("Too complex, sigh");
        }

    public:
        void initialize() {
            typename BSplineHomogeneous::ControlPoints ctrl_pts(
                    Base::m_control_points.rows(), _dim+1);
            ctrl_pts.template leftCols<_dim>() =
                Base::m_control_points.array().colwise() * m_weights.array();
            ctrl_pts.template rightCols<1>() = m_weights;

            m_bspline_homogeneous.set_control_points(std::move(ctrl_pts));
            m_bspline_homogeneous.set_knots(Base::m_knots);
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
        BSplineHomogeneous m_bspline_homogeneous;
        WeightVector m_weights;
};

}
