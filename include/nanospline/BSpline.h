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
            assert(Base::in_domain(t));
            Base::validate_curve();
            const int p = Base::get_degree();
            const int k = Base::locate_span(t);
            assert(p >= 0);
            assert(Base::m_knots.rows() ==
                    Base::m_control_points.rows() + p + 1);
            assert(Base::m_knots[k] <= t);
            assert(Base::m_knots[k+1] >= t);

            ControlPoints ctrl_pts(p+1, _dim);
            for (int i=0; i<=p; i++) {
                ctrl_pts.row(i) = Base::m_control_points.row(i+k-p);
            }

            deBoor(t, p, k, ctrl_pts);
            return ctrl_pts.row(p);
        }

        Scalar inverse_evaluate(const Point& p) const override {
            throw not_implemented_error("Too complex, sigh");
        }

        Point evaluate_derivative(Scalar t) const override {
            assert(Base::in_domain(t));
            Base::validate_curve();
            const int p = Base::get_degree();
            const int k = Base::locate_span(t);
            assert(p >= 0);
            assert(Base::m_knots.rows() ==
                    Base::m_control_points.rows() + p + 1);
            assert(Base::m_knots[k] <= t);
            assert(Base::m_knots[k+1] >= t);

            if (p == 0) return Point::Zero();

            ControlPoints ctrl_pts(p, _dim);
            for(int i=0; i<p; i++) {
                const Scalar diff = Base::m_knots[i+k+1] - Base::m_knots[i+k-p+1];
                Scalar alpha = 0.0;
                if (diff > 0) {
                    alpha = p / diff;
                }
                ctrl_pts.row(i) = alpha * (
                        Base::m_control_points.row(i+k-p+1) -
                        Base::m_control_points.row(i+k-p));
            }

            deBoor(t, p-1, k, ctrl_pts);
            return ctrl_pts.row(p-1);
        }

        Point evaluate_2nd_derivative(Scalar t) const override {
            assert(Base::in_domain(t));
            Base::validate_curve();
            const int p = Base::get_degree();
            const int k = Base::locate_span(t);
            assert(p >= 0);
            assert(Base::m_knots.rows() ==
                    Base::m_control_points.rows() + p + 1);
            assert(Base::m_knots[k] <= t);
            assert(Base::m_knots[k+1] >= t);

            if (p <= 1) return Point::Zero();

            ControlPoints ctrl_pts(p, _dim);

            // First derivative control pts.
            for(int i=0; i<p; i++) {
                const Scalar diff = Base::m_knots[i+k+1] - Base::m_knots[i+k-p+1];
                Scalar alpha = 0.0;
                if (diff > 0) {
                    alpha = p / diff;
                }
                ctrl_pts.row(i) = alpha * (
                        Base::m_control_points.row(i+k-p+1) -
                        Base::m_control_points.row(i+k-p));
            }

            // Second derivative control pts.
            for (int i=0; i<p-1; i++) {
                const Scalar diff = Base::m_knots[i+k+1] - Base::m_knots[i+k-p+2];
                Scalar alpha = 0.0;
                if (diff > 0) {
                    alpha = (p-1) / diff;
                }
                ctrl_pts.row(i) = alpha * (ctrl_pts.row(i+1) - ctrl_pts.row(i));
            }

            deBoor(t, p-2, k, ctrl_pts);
            return ctrl_pts.row(p-2);
        }

    private:
        template<typename Derived>
        void deBoor(Scalar t, int p, int k,
                Eigen::PlainObjectBase<Derived>& ctrl_pts) const {
            assert(ctrl_pts.rows() >= p+1);

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
        }
};

}
