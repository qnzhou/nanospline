#pragma once

#include <algorithm>

#include <Eigen/Core>
#include <nanospline/Exceptions.h>
#include <nanospline/BezierBase.h>

namespace nanospline {

template<typename _Scalar, int _dim=2, int _order=3, bool _generic=_order<0 >
class Bezier : public BezierBase<_Scalar, _dim, _order, _generic> {
    public:
        using Base = BezierBase<_Scalar, _dim, _order, _generic>;
        using Scalar = typename Base::Scalar;
        using Point = typename Base::Point;
        using ControlPoints = typename Base::ControlPoints;

    public:
        Point evaluate(Scalar t) const override {
            assert(t>=0.0 && t <= 1.0);
            const int order = Base::m_control_points.rows()-1;
            if (order < 0) {
                throw invalid_setting_error("Negative Bezier order.");
            }

            ControlPoints pts[2];
            pts[0] = Base::m_control_points;
            pts[1].resize(order+1, _dim);

            for (int i=0; i<order; i++) {
                const auto N = order-i;
                const auto& curr_ctrl_pts = pts[i%2];
                auto& next_ctrl_pts = pts[(i+1)%2];

                for (int j=0; j<N; j++) {
                    next_ctrl_pts.row(j) =
                        (1.0-t) * curr_ctrl_pts.row(j) +
                        t * curr_ctrl_pts.row(j+1);
                }
            }

            return pts[order%2].row(0);
        }

        Scalar inverse_evaluate(const Point& p) const override {
            throw not_implemented_error("Too complex, sigh");
        }
};

template<typename _Scalar, int _dim>
class Bezier<_Scalar, _dim, 0, false> : public BezierBase<_Scalar, _dim, 0, false> {
    public:
        using Base = BezierBase<_Scalar, _dim, 0, false>;
        using Scalar = typename Base::Scalar;
        using Point = typename Base::Point;
        using ControlPoints = typename Base::ControlPoints;

    public:
        Point evaluate(Scalar t) const override {
            return Base::m_control_points;
        }

        Scalar inverse_evaluate(const Point& p) const override {
            return 0.0;
        }
};

template<typename _Scalar, int _dim>
class Bezier<_Scalar, _dim, 1, false> : public BezierBase<_Scalar, _dim, 1, false> {
    public:
        using Base = BezierBase<_Scalar, _dim, 1, false>;
        using Scalar = typename Base::Scalar;
        using Point = typename Base::Point;
        using ControlPoints = typename Base::ControlPoints;

    public:
        Point evaluate(Scalar t) const override {
            return (1.0-t) * Base::m_control_points.row(0) +
                t * Base::m_control_points.row(1);
        }

        Scalar inverse_evaluate(const Point& p) const override {
            Point e = Base::m_control_points.row(1) - Base::m_control_points.row(0);
            Scalar t = (p - Base::m_control_points.row(0)).dot(e) / e.squaredNorm();
            return std::max<Scalar>(std::min<Scalar>(t, 1.0), 0.0);
        }
};

template<typename _Scalar, int _dim>
class Bezier<_Scalar, _dim, 2, false> : public BezierBase<_Scalar, _dim, 2, false> {
    public:
        using Base = BezierBase<_Scalar, _dim, 2, false>;
        using Scalar = typename Base::Scalar;
        using Point = typename Base::Point;
        using ControlPoints = typename Base::ControlPoints;

    public:
        Point evaluate(Scalar t) const override {
            const Point p0 = (1.0-t) * Base::m_control_points.row(0) +
                t * Base::m_control_points.row(1);
            const Point p1 = (1.0-t) * Base::m_control_points.row(1) +
                t * Base::m_control_points.row(2);
            return (1.0-t) * p0 + t * p1;
        }

        Scalar inverse_evaluate(const Point& p) const override {
            throw not_implemented_error("Too complex, sigh");
        }
};

template<typename _Scalar, int _dim>
class Bezier<_Scalar, _dim, 3, false> : public BezierBase<_Scalar, _dim, 3, false> {
    public:
        using Base = BezierBase<_Scalar, _dim, 3, false>;
        using Scalar = typename Base::Scalar;
        using Point = typename Base::Point;
        using ControlPoints = typename Base::ControlPoints;

    public:
        Point evaluate(Scalar t) const override {
            const Point q0 = (1.0-t) * Base::m_control_points.row(0) +
                t * Base::m_control_points.row(1);
            const Point q1 = (1.0-t) * Base::m_control_points.row(1) +
                t * Base::m_control_points.row(2);
            const Point q2 = (1.0-t) * Base::m_control_points.row(2) +
                t * Base::m_control_points.row(3);

            const Point p0 = (1.0-t) * q0 + t * q1;
            const Point p1 = (1.0-t) * q1 + t * q2;
            return (1.0-t) * p0 + t * p1;
        }

        Scalar inverse_evaluate(const Point& p) const override {
            throw not_implemented_error("Too complex, sigh");
        }
};


}
