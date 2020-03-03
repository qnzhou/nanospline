#pragma once

#include <algorithm>

#include <Eigen/Core>
#include <nanospline/Exceptions.h>
#include <nanospline/BezierBase.h>
#include <nanospline/internal/auto_inflection_Bezier.h>
#include <nanospline/internal/auto_match_tangent_Bezier.h>

namespace nanospline {

template<typename _Scalar, int _dim=2, int _degree=3, bool _generic=_degree<0 >
class Bezier : public BezierBase<_Scalar, _dim, _degree, _generic> {
    public:
        using Base = BezierBase<_Scalar, _dim, _degree, _generic>;
        using Scalar = typename Base::Scalar;
        using Point = typename Base::Point;
        using ControlPoints = typename Base::ControlPoints;

    public:
        Point evaluate(Scalar t) const override {
            const auto control_pts = deBoor(t, Base::get_degree());
            return control_pts.row(0);
        }

        Scalar inverse_evaluate(const Point& p) const override {
            throw not_implemented_error("Too complex, sigh");
        }

        Point evaluate_derivative(Scalar t) const override {
            const auto degree = Base::get_degree();
            assert(degree >= 0);
            if (degree == 0) {
                return Point::Zero();
            } else {
                const auto control_pts = deBoor(t, degree-1);
                return (control_pts.row(1) - control_pts.row(0))*degree;
            }
        }

        Point evaluate_2nd_derivative(Scalar t) const override {
            const auto degree = Base::get_degree();
            assert(degree >= 0);
            if (degree <= 1) {
                return Point::Zero();
            } else {
                const auto control_pts = deBoor(t, degree-2);
                return (control_pts.row(2) + control_pts.row(0) - 2 * control_pts.row(1))
                    * degree * (degree-1);
            }
        }

        std::vector<Scalar> compute_inflections (
                const Scalar lower,
                const Scalar upper) const override final {
            if (_dim != 2) {
                throw std::runtime_error(
                        "Inflection computation is for 2D curves only");
            }
            auto res = nanospline::internal::compute_Bezier_inflections(
                    Base::m_control_points, lower, upper);

            std::sort(res.begin(), res.end());
            res.erase(std::unique(res.begin(), res.end()), res.end());

            return res;
        }

        std::vector<Scalar> reduce_turning_angle(
                const Scalar lower,
                const Scalar upper) const override final {
            constexpr Scalar tol = static_cast<Scalar>(1e-8);
            if (_dim != 2) {
                throw std::runtime_error(
                        "Turning angle reduction is for 2D curves only");
            }

            auto tan0 = evaluate_derivative(lower);
            auto tan1 = evaluate_derivative(upper);

            if (tan0.norm() < tol || tan1.norm() < tol){
                std::vector<Scalar> res;
                res.push_back((lower + upper) / 2);
                return res;
            }

            tan0 = tan0 / tan0.norm();
            tan1 = tan1 / tan1.norm();

            const Eigen::Matrix<Scalar, 2, 1> ave_tangent(
                    -(tan0[1]+tan1[1])/2,
                    (tan0[0]+tan1[0])/2);

            const auto degree = Base::get_degree();
            auto res = nanospline::internal::match_tangent_bezier(
                    Base::m_control_points, degree, ave_tangent, lower, upper);

            std::sort(res.begin(), res.end());
            res.erase(std::unique(res.begin(), res.end()), res.end());

            return res;
        }

    private:
        ControlPoints deBoor(Scalar t, int num_recurrsions) const {
            const auto degree = Base::get_degree();
            if (num_recurrsions < 0 || num_recurrsions > degree) {
                throw invalid_setting_error(
                        "Number of de Boor recurrsion cannot exceeds degree");
            }

            if (num_recurrsions == 0) {
                return Base::m_control_points;
            } else {
                ControlPoints ctrl_pts = deBoor(t, num_recurrsions-1);
                assert(ctrl_pts.rows() >= degree+1-num_recurrsions);
                for (int i=0; i<degree+1-num_recurrsions; i++) {
                    ctrl_pts.row(i) = (1.0-t) * ctrl_pts.row(i) + t * ctrl_pts.row(i+1);
                }
                return ctrl_pts;
            }
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

        Point evaluate_derivative(Scalar t) const override {
            return Point::Zero();
        }

        Point evaluate_2nd_derivative(Scalar t) const override {
            return Point::Zero();
        }

        std::vector<Scalar> compute_inflections(
                const Scalar lower=0.0,
                const Scalar upper=1.0) const override final {
            return {};
        }

        std::vector<Scalar> reduce_turning_angle(
                const Scalar lower,
                const Scalar upper) const override final {
            return {};
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

        Point evaluate_derivative(Scalar t) const override {
            return Base::m_control_points.row(1) - Base::m_control_points.row(0);
        }

        Point evaluate_2nd_derivative(Scalar t) const override {
            return Point::Zero();
        }

        std::vector<Scalar> compute_inflections(
                const Scalar lower=0.0,
                const Scalar upper=1.0) const override final {
            return {};
        }

        std::vector<Scalar> reduce_turning_angle(
                const Scalar lower,
                const Scalar upper) const override final {
            return {};
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

        Point evaluate_derivative(Scalar t) const override {
            const Point p0 = (1.0-t) * Base::m_control_points.row(0) +
                t * Base::m_control_points.row(1);
            const Point p1 = (1.0-t) * Base::m_control_points.row(1) +
                t * Base::m_control_points.row(2);
            return (p1-p0) * 2;
        }

        Point evaluate_2nd_derivative(Scalar t) const override {
            const auto& ctrl_pts = Base::m_control_points;
            return 2 * (ctrl_pts.row(0) + ctrl_pts.row(2) - 2 * ctrl_pts.row(1));
        }

        std::vector<Scalar> compute_inflections(
                const Scalar lower=0.0,
                const Scalar upper=1.0) const override final {
            return {};
        }

        std::vector<Scalar> reduce_turning_angle(
                const Scalar lower,
                const Scalar upper) const override final {
            constexpr Scalar tol = static_cast<Scalar>(1e-8);
            if (_dim != 2) {
                throw std::runtime_error(
                        "Turning angle reduction is for 2D curves only");
            }

            auto tan0 = evaluate_derivative(lower);
            auto tan1 = evaluate_derivative(upper);

            if (tan0.norm() < tol || tan1.norm() < tol){
                std::vector<Scalar> res;
                res.push_back((lower + upper) / 2);
                return res;
            }

            tan0 = tan0 / tan0.norm();
            tan1 = tan1 / tan1.norm();

            const Eigen::Matrix<Scalar, 2, 1> ave_tangent(
                    -(tan0[1]+tan1[1])/2,
                    (tan0[0]+tan1[0])/2);

            auto res = nanospline::internal::match_tangent_Bezier_degree_2(
                    Base::m_control_points(0, 0),
                    Base::m_control_points(0, 1),
                    Base::m_control_points(1, 0),
                    Base::m_control_points(1, 1),
                    Base::m_control_points(2, 0),
                    Base::m_control_points(2, 1),
                    ave_tangent, lower, upper);

            std::sort(res.begin(), res.end());
            res.erase(std::unique(res.begin(), res.end()), res.end());

            return res;
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

        Point evaluate_derivative(Scalar t) const override {
            const Point q0 = (1.0-t) * Base::m_control_points.row(0) +
                t * Base::m_control_points.row(1);
            const Point q1 = (1.0-t) * Base::m_control_points.row(1) +
                t * Base::m_control_points.row(2);
            const Point q2 = (1.0-t) * Base::m_control_points.row(2) +
                t * Base::m_control_points.row(3);

            const Point p0 = (1.0-t) * q0 + t * q1;
            const Point p1 = (1.0-t) * q1 + t * q2;

            return (p1-p0)*3;
        }

        Point evaluate_2nd_derivative(Scalar t) const override {
            const Point q0 = (1.0-t) * Base::m_control_points.row(0) +
                t * Base::m_control_points.row(1);
            const Point q1 = (1.0-t) * Base::m_control_points.row(1) +
                t * Base::m_control_points.row(2);
            const Point q2 = (1.0-t) * Base::m_control_points.row(2) +
                t * Base::m_control_points.row(3);
            return 6 * (q0+q2-2*q1);
        }

        std::vector<Scalar> compute_inflections(
                const Scalar lower,
                const Scalar upper) const override final {
            auto res = nanospline::internal::compute_Bezier_degree_3_inflections(
                    Base::m_control_points(0, 0),
                    Base::m_control_points(0, 1),
                    Base::m_control_points(1, 0),
                    Base::m_control_points(1, 1),
                    Base::m_control_points(2, 0),
                    Base::m_control_points(2, 1),
                    Base::m_control_points(3, 0),
                    Base::m_control_points(3, 1),
                    lower, upper);

            std::sort(res.begin(), res.end());
            res.erase(std::unique(res.begin(), res.end()), res.end());

            return res;
        }

        std::vector<Scalar> reduce_turning_angle(
                const Scalar lower,
                const Scalar upper) const override final {
            constexpr Scalar tol = static_cast<Scalar>(1e-8);
            if (_dim != 2) {
                throw std::runtime_error(
                        "Turning angle reduction is for 2D curves only");
            }

            auto tan0 = evaluate_derivative(lower);
            auto tan1 = evaluate_derivative(upper);

            if (tan0.norm() < tol || tan1.norm() < tol){
                std::vector<Scalar> res;
                res.push_back((lower + upper) / 2);
                return res;
            }

            tan0 = tan0 / tan0.norm();
            tan1 = tan1 / tan1.norm();

            const Eigen::Matrix<Scalar, 2, 1> ave_tangent(
                    -(tan0[1]+tan1[1])/2,
                    (tan0[0]+tan1[0])/2);

            auto res = nanospline::internal::match_tangent_Bezier_degree_3(
                    Base::m_control_points(0, 0),
                    Base::m_control_points(0, 1),
                    Base::m_control_points(1, 0),
                    Base::m_control_points(1, 1),
                    Base::m_control_points(2, 0),
                    Base::m_control_points(2, 1),
                    Base::m_control_points(3, 0),
                    Base::m_control_points(3, 1),
                    ave_tangent, lower, upper);

            std::sort(res.begin(), res.end());
            res.erase(std::unique(res.begin(), res.end()), res.end());

            return res;
        }
};


}
