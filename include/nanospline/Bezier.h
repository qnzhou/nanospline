#pragma once

#include <algorithm>

#include <Eigen/Core>
#include <nanospline/Exceptions.h>

namespace nanospline {

template<typename _Scalar, int _dim=2, int _order=3, bool _generic=_order<0 >
class Bezier {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        using Scalar = _Scalar;
        using Point = Eigen::Matrix<Scalar, 1, _dim>;
        using ControlPoints = Eigen::Matrix<Scalar, _generic?Eigen::Dynamic:_order+1, _dim>;

    public:
        template<typename Derived>
        void set_control_points(const Eigen::PlainObjectBase<Derived>& ctrl_pts) {
            m_control_points = ctrl_pts;
        }

        template<typename Derived>
        void set_control_points(Eigen::PlainObjectBase<Derived>&& ctrl_pts) {
            m_control_points.swap(ctrl_pts);
        }

    public:
        Point evaluate(Scalar t) const {
            assert(t>=0.0 && t <= 1.0);
            const int order = m_control_points.rows()-1;
            if (order < 0) {
                throw invalid_setting_error("Negative Bezier order.");
            }

            ControlPoints pts[2];
            pts[0] = m_control_points;
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

        Scalar inverse_evaluate(const Point& p) const {
            throw not_implemented_error("Too complex, sigh");
        }

    private:
        ControlPoints m_control_points;
};

template<typename _Scalar, int _dim>
class Bezier<_Scalar, _dim, 0, false> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        using Scalar = _Scalar;
        using Point = Eigen::Matrix<Scalar, 1, _dim>;
        using ControlPoints = Eigen::Matrix<Scalar, 1, _dim>;

    public:
        template<typename Derived>
        void set_control_points(const Eigen::PlainObjectBase<Derived>& ctrl_pts) {
            m_control_points = ctrl_pts;
        }

        template<typename Derived>
        void set_control_points(Eigen::PlainObjectBase<Derived>&& ctrl_pts) {
            m_control_points.swap(ctrl_pts);
        }

    public:
        Point evaluate(Scalar t) const {
            return m_control_points;
        }

        Scalar inverse_evaluate(const Point& p) const {
            return 0.0;
        }

    private:
        ControlPoints m_control_points;
};

template<typename _Scalar, int _dim>
class Bezier<_Scalar, _dim, 1, false> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        using Scalar = _Scalar;
        using Point = Eigen::Matrix<Scalar, 1, _dim>;
        using ControlPoints = Eigen::Matrix<Scalar, 2, _dim>;

    public:
        template<typename Derived>
        void set_control_points(const Eigen::PlainObjectBase<Derived>& ctrl_pts) {
            m_control_points = ctrl_pts;
        }

        template<typename Derived>
        void set_control_points(Eigen::PlainObjectBase<Derived>&& ctrl_pts) {
            m_control_points.swap(ctrl_pts);
        }

    public:
        Point evaluate(Scalar t) const {
            return (1.0-t) * m_control_points.row(0) +
                t * m_control_points.row(1);
        }

        Scalar inverse_evaluate(const Point& p) const {
            Point e = m_control_points.row(1) - m_control_points.row(0);
            Scalar t = (p - m_control_points.row(0)).dot(e) / e.squaredNorm();
            return std::max<Scalar>(std::min<Scalar>(t, 1.0), 0.0);
        }

    private:
        ControlPoints m_control_points;
};

template<typename _Scalar, int _dim>
class Bezier<_Scalar, _dim, 2, false> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        using Scalar = _Scalar;
        using Point = Eigen::Matrix<Scalar, 1, _dim>;
        using ControlPoints = Eigen::Matrix<Scalar, 3, _dim>;

    public:
        template<typename Derived>
        void set_control_points(const Eigen::PlainObjectBase<Derived>& ctrl_pts) {
            m_control_points = ctrl_pts;
        }

        template<typename Derived>
        void set_control_points(Eigen::PlainObjectBase<Derived>&& ctrl_pts) {
            m_control_points.swap(ctrl_pts);
        }

    public:
        Point evaluate(Scalar t) const {
            const Point p0 = (1.0-t) * m_control_points.row(0) +
                t * m_control_points.row(1);
            const Point p1 = (1.0-t) * m_control_points.row(1) +
                t * m_control_points.row(2);
            return (1.0-t) * p0 + t * p1;
        }

        Scalar inverse_evaluate(const Point& p) const {
            throw not_implemented_error("Too complex, sigh");
        }

    private:
        ControlPoints m_control_points;
};

template<typename _Scalar, int _dim>
class Bezier<_Scalar, _dim, 3, false> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        using Scalar = _Scalar;
        using Point = Eigen::Matrix<Scalar, 1, _dim>;
        using ControlPoints = Eigen::Matrix<Scalar, 4, _dim>;

    public:
        template<typename Derived>
        void set_control_points(const Eigen::PlainObjectBase<Derived>& ctrl_pts) {
            m_control_points = ctrl_pts;
        }

        template<typename Derived>
        void set_control_points(Eigen::PlainObjectBase<Derived>&& ctrl_pts) {
            m_control_points.swap(ctrl_pts);
        }

    public:
        Point evaluate(Scalar t) const {
            const Point q0 = (1.0-t) * m_control_points.row(0) +
                t * m_control_points.row(1);
            const Point q1 = (1.0-t) * m_control_points.row(1) +
                t * m_control_points.row(2);
            const Point q2 = (1.0-t) * m_control_points.row(2) +
                t * m_control_points.row(3);

            const Point p0 = (1.0-t) * q0 + t * q1;
            const Point p1 = (1.0-t) * q1 + t * q2;
            return (1.0-t) * p0 + t * p1;
        }

        Scalar inverse_evaluate(const Point& p) const {
            throw not_implemented_error("Too complex, sigh");
        }

    private:
        ControlPoints m_control_points;
};


}
