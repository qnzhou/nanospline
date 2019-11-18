#pragma once

#include <Eigen/Core>

#include <nanospline/Exceptions.h>
#include <nanospline/BSplineBase.h>

namespace nanospline {

template<typename _Scalar, int _dim=3, int _degree=3, bool _generic=_degree<0 >
class BSpline : public BSplineBase<_Scalar, _dim, _degree, _generic> {
    public:
        using Base = BSplineBase<_Scalar, _dim, _degree, _generic>;
        using Scalar = typename Base::Scalar;
        using Point = typename Base::Point;
        using ControlPoints = typename Base::ControlPoints;
        using KnotVector = typename Base::KnotVector;

    public:
        Point evaluate(Scalar t) const override {
            const int k = Base::locate_span(t);
            const int p = Base::get_degree();
            assert(p >= 0);
            assert(Base::m_knots.rows() == 
                    Base::m_control_points.rows() + p + 1);
            assert(Base::m_knots[k] <= t);
            assert(Base::m_knots[k+1] >= t);

            ControlPoints ctrl_pts(p+1, _dim);
            for (int i=0; i<=p; i++) {
                ctrl_pts.row(i) = Base::m_control_points.row(i+k-p);
            }

            for (int r=1; r<=p; r++) {
                for (int j=p; j>=r; j--) {
                    const Scalar diff =
                        Base::m_knots[j+1+k-r] - Base::m_knots[j+k-p];
                    Scalar alpha = 0.0;
                    if (diff > 0) {
                        alpha = (t - Base::m_knots[j+k-p]) / diff;
                    }

                    ctrl_pts.row(j) = (1.0-alpha) * ctrl_pts.row(j-1) +
                        alpha * ctrl_pts.row(j);
                }
            }

            return ctrl_pts.row(p);
        }

        Scalar inverse_evaluate(const Point& p) const override {
            throw not_implemented_error("Too complex, sigh");
        }
};

}
