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

        void insert_knot(Scalar t, int multiplicity=1) {
            assert(Base::in_domain(t));
            const int r = multiplicity;
            const int p = Base::get_degree();
            const int k = Base::locate_span(t);
            const int s = (t == Base::m_knots[k]) ? Base::get_multiplicity(k):0;
            assert(k>=p);
            assert(r+s<=p);

            const int n = Base::m_control_points.rows()-1;
            const int m = Base::m_knots.rows()-1;

            KnotVector knots_new(m+r+1, 1);
            knots_new.segment(0, k+1) = Base::m_knots.segment(0, k+1);
            knots_new.segment(k+1, r).setConstant(t);
            knots_new.segment(k+1+r, m-k-1) = Base::m_knots.segment(k+1,m-k-1);

            ControlPoints ctrl_pts_new(n+r+1, _dim);
            ctrl_pts_new.topRows(k-p+1) = Base::m_control_points.topRows(k-p+1);
            ctrl_pts_new.bottomRows(n-k-s+1) = Base::m_control_points.bottomRows(n-k+s+1);

            ControlPoints Rw = Base::m_control_points.block(k-p, 0, p-s+1, _dim);
            for (int j=1; j<=r; j++) {
                int L = k-p+j;
                for (int i=0; i<=p-j-s; i++) {
                    const Scalar diff = Base::m_knots[i+k+1] - Base::m_knots[L+i];
                    Scalar alpha = 0.0;
                    if (diff > 0) {
                        alpha = (t - Base::m_knots[L+i]) / diff;
                    }
                    Rw.row(i) = alpha * Rw.row(i+1) + (1.0-alpha) * Rw.row(i);
                }
                ctrl_pts_new.row(L) = Rw.row(0);
                ctrl_pts_new.row(k+r-j-s) = Rw.row(p-j-s);
            }

            for (int i=k-p+r; i<k-s; i++) {
                ctrl_pts_new.row(i) = Rw.row(i-k+p-r);
            }

            Base::m_control_points.swap(ctrl_pts_new);
            Base::m_knots.swap(knots_new);
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
