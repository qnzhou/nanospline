#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include <Eigen/Core>
#include <nanospline/Exceptions.h>
#include <nanospline/BezierBase.h>
#include <nanospline/internal/auto_inflection_Bezier.h>
#include <nanospline/internal/auto_match_tangent_Bezier.h>
#include <nanospline/internal/auto_singularity.h>

namespace nanospline {

template<typename _Scalar, int _dim=2, int _degree=3, bool _generic=_degree<0 >
class Bezier final : public BezierBase<_Scalar, _dim, _degree, _generic> {
    public:
        using Base = BezierBase<_Scalar, _dim, _degree, _generic>;
        using ThisType = Bezier<_Scalar, _dim, _degree, _generic>;
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
                const Scalar upper) const override {
            if (_dim != 2) {
                throw std::runtime_error(
                        "Inflection computation is for 2D curves only");
            }
            const auto degree = Base::get_degree();
            if (degree <= 2) {
                return {};
            }
            std::vector<Scalar> res;

            try {
                res = nanospline::internal::compute_Bezier_inflections(
                        Base::m_control_points, lower, upper);
            } catch (infinite_root_error e) {
                // Infinitely many inflections.
                res.clear();
            }

            std::sort(res.begin(), res.end());
            res.erase(std::unique(res.begin(), res.end()), res.end());

            return res;
        }

        Scalar get_turning_angle(Scalar t0, Scalar t1) const override {
            if (_dim != 2) {
                throw invalid_setting_error(
                        "Turning angle is for 2D only");
            }
            auto segment = subcurve(t0, t1);
            const auto& ctrl_pts = segment.get_control_points();
            const auto num_ctrl_pts = ctrl_pts.rows();
            Scalar theta = 0;
            for (int i=1; i+1<num_ctrl_pts; i++) {
                const Point v0 = ctrl_pts.row(i) - ctrl_pts.row(i-1);
                const Point v1 = ctrl_pts.row(i+1) - ctrl_pts.row(i);
                theta += std::atan2(
                        v0[0]*v1[1] - v0[1]*v1[0],
                        v0[0]*v1[0] + v0[1]*v1[1]);
            }
            return theta;
        }

        std::vector<Scalar> reduce_turning_angle(
                const Scalar lower,
                const Scalar upper) const override {
            constexpr Scalar tol = static_cast<Scalar>(1e-8);
            if (_dim != 2) {
                throw std::runtime_error(
                        "Turning angle reduction is for 2D curves only");
            }
            const auto degree = Base::get_degree();
            if (degree < 2) {
                return {};
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

            std::vector<Scalar> res;
            try {
                res = nanospline::internal::match_tangent_bezier(
                        Base::m_control_points, degree, ave_tangent, lower, upper);
            } catch (infinite_root_error) {
                res.clear();
            }

            std::sort(res.begin(), res.end());
            res.erase(std::unique(res.begin(), res.end()), res.end());

            return res;
        }

        std::vector<Scalar> compute_singularities(
                const Scalar lower=0.0,
                const Scalar upper=1.0) const override {
            std::vector<Scalar> res = nanospline::internal::compute_singularities(
                    Base::m_control_points, lower, upper);

            std::sort(res.begin(), res.end());
            res.erase(std::unique(res.begin(), res.end()), res.end());

            return res;
        }

        /**
         * Split a curve in halves at t.
         */
        std::array<ThisType, 2> split(Scalar t) const {
            assert(t >= 0);
            assert(t <= 1);
            // Copy intentionally.
            auto ctrl_pts = Base::get_control_points();
            const auto d = Base::get_degree();

            ControlPoints ctrl_pts_1(d+1, _dim);
            ControlPoints ctrl_pts_2(d+1, _dim);

            for (int i=0; i<=d; i++) {
                ctrl_pts_1.row(i) = ctrl_pts.row(0);
                ctrl_pts_2.row(d-i) = ctrl_pts.row(d-i);
                for (int j=0; j<d-i; j++) {
                    ctrl_pts.row(j) = (1.0-t) * ctrl_pts.row(j) + t * ctrl_pts.row(j+1);
                }
            }

            std::array<ThisType, 2> results;
            results[0].set_control_points(std::move(ctrl_pts_1));
            results[1].set_control_points(std::move(ctrl_pts_2));

            return results;
        }

        /**
         * Extract a subcurve in range [t0, t1].
         */
        ThisType subcurve(Scalar t0, Scalar t1) const {
            if (t0 > t1) {
                throw invalid_setting_error("t0 must be smaller than t1");
            }
            if (t0 < 0 || t0 > 1 || t1 < 0 || t1 > 1) {
                throw invalid_setting_error("Invalid range");
            }
            return split(t0)[1].split(t1)[0];
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
class Bezier<_Scalar, _dim, 0, false> final : public BezierBase<_Scalar, _dim, 0, false> {
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
                const Scalar upper=1.0) const override {
            return {};
        }

        Scalar get_turning_angle(Scalar t0, Scalar t1) const override {
            return 0;
        }

        std::vector<Scalar> reduce_turning_angle(
                const Scalar lower,
                const Scalar upper) const override {
            return {};
        }

        std::vector<Scalar> compute_singularities(
                const Scalar lower,
                const Scalar upper) const override {
            return {};
        }
};

template<typename _Scalar, int _dim>
class Bezier<_Scalar, _dim, 1, false> final : public BezierBase<_Scalar, _dim, 1, false> {
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
                const Scalar upper=1.0) const override {
            return {};
        }

        Scalar get_turning_angle(Scalar t0, Scalar t1) const override {
            return 0;
        }

        std::vector<Scalar> reduce_turning_angle(
                const Scalar lower,
                const Scalar upper) const override {
            return {};
        }

        std::vector<Scalar> compute_singularities(
                const Scalar lower,
                const Scalar upper) const override {
            return {};
        }
};

template<typename _Scalar, int _dim>
class Bezier<_Scalar, _dim, 2, false> final : public BezierBase<_Scalar, _dim, 2, false> {
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
                const Scalar upper=1.0) const override {
            return {};
        }

        Scalar get_turning_angle(Scalar t0, Scalar t1) const override {
            if (_dim != 2) {
                throw invalid_setting_error(
                        "Turning angle is for 2D only");
            }

            const Point v0 = evaluate_derivative(t0);
            const Point v1 = evaluate_derivative(t1);
            return std::atan2(
                    v0[0]*v1[1] - v0[1]*v1[0],
                    v0[0]*v1[0] + v0[1]*v1[1]);
        }

        std::vector<Scalar> reduce_turning_angle(
                const Scalar lower,
                const Scalar upper) const override {
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

            std::vector<Scalar> res;
            try {
                res = nanospline::internal::match_tangent_Bezier_degree_2(
                        Base::m_control_points(0, 0),
                        Base::m_control_points(0, 1),
                        Base::m_control_points(1, 0),
                        Base::m_control_points(1, 1),
                        Base::m_control_points(2, 0),
                        Base::m_control_points(2, 1),
                        ave_tangent, lower, upper);
            } catch (infinite_root_error) {
                res.clear();
            }

            std::sort(res.begin(), res.end());
            res.erase(std::unique(res.begin(), res.end()), res.end());

            return res;
        }

        std::vector<Scalar> compute_singularities(
                const Scalar lower,
                const Scalar upper) const override {
            auto res = nanospline::internal::compute_degree_2_singularities(
                    Base::m_control_points(0, 0),
                    Base::m_control_points(0, 1),
                    Base::m_control_points(1, 0),
                    Base::m_control_points(1, 1),
                    Base::m_control_points(2, 0),
                    Base::m_control_points(2, 1),
                    lower, upper);

            std::sort(res.begin(), res.end());
            res.erase(std::unique(res.begin(), res.end()), res.end());

            return res;
        }
};

template<typename _Scalar, int _dim>
class Bezier<_Scalar, _dim, 3, false> final : public BezierBase<_Scalar, _dim, 3, false> {
    public:
        using Base = BezierBase<_Scalar, _dim, 3, false>;
        using ThisType = Bezier<_Scalar, _dim, 3, false>;
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
                const Scalar upper) const override {
            std::vector<Scalar> res;
            try {
                res = nanospline::internal::compute_Bezier_degree_3_inflections(
                        Base::m_control_points(0, 0),
                        Base::m_control_points(0, 1),
                        Base::m_control_points(1, 0),
                        Base::m_control_points(1, 1),
                        Base::m_control_points(2, 0),
                        Base::m_control_points(2, 1),
                        Base::m_control_points(3, 0),
                        Base::m_control_points(3, 1),
                        lower, upper);
            } catch (infinite_root_error) {
                res.clear();
            }

            std::sort(res.begin(), res.end());
            res.erase(std::unique(res.begin(), res.end()), res.end());

            return res;
        }

        Scalar get_turning_angle(Scalar t0, Scalar t1) const override {
            if (_dim != 2) {
                throw invalid_setting_error(
                        "Turning angle is for 2D only");
            }

            const auto piece = subcurve(t0, t1);
            const auto& ctrl_pts = piece.get_control_points();

            const Point v0 = ctrl_pts.row(1) - ctrl_pts.row(0);
            const Point v1 = ctrl_pts.row(2) - ctrl_pts.row(1);
            const Point v2 = ctrl_pts.row(3) - ctrl_pts.row(2);

            return std::atan2(
                    v0[0]*v1[1] - v0[1]*v1[0],
                    v0[0]*v1[0] + v0[1]*v1[1]) +
                std::atan2(
                    v1[0]*v2[1] - v1[1]*v2[0],
                    v1[0]*v2[0] + v1[1]*v2[1]);
        }

        std::vector<Scalar> reduce_turning_angle(
                const Scalar lower,
                const Scalar upper) const override {
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

            std::vector<Scalar> res;
            try {
                res = nanospline::internal::match_tangent_Bezier_degree_3(
                        Base::m_control_points(0, 0),
                        Base::m_control_points(0, 1),
                        Base::m_control_points(1, 0),
                        Base::m_control_points(1, 1),
                        Base::m_control_points(2, 0),
                        Base::m_control_points(2, 1),
                        Base::m_control_points(3, 0),
                        Base::m_control_points(3, 1),
                        ave_tangent, lower, upper);
            } catch (infinite_root_error) {
                res.clear();
            }

            std::sort(res.begin(), res.end());
            res.erase(std::unique(res.begin(), res.end()), res.end());

            return res;
        }

        std::vector<Scalar> compute_singularities(
                const Scalar lower,
                const Scalar upper) const override {
            std::vector<Scalar> res = nanospline::internal::compute_degree_3_singularities(
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

    public:
        std::array<ThisType, 2> split(Scalar t) const {
            assert(t >= 0);
            assert(t <= 1);
            // Copy intentionally.
            auto ctrl_pts = Base::get_control_points();

            ControlPoints ctrl_pts_1(4, _dim);
            ControlPoints ctrl_pts_2(4, _dim);

            for (int i=0; i<=3; i++) {
                ctrl_pts_1.row(i) = ctrl_pts.row(0);
                ctrl_pts_2.row(3-i) = ctrl_pts.row(3-i);
                for (int j=0; j<3-i; j++) {
                    ctrl_pts.row(j) = (1.0-t) * ctrl_pts.row(j) + t * ctrl_pts.row(j+1);
                }
            }

            std::array<ThisType, 2> results;
            results[0].set_control_points(std::move(ctrl_pts_1));
            results[1].set_control_points(std::move(ctrl_pts_2));

            return results;
        }

        /**
         * Extract a subcurve in range [t0, t1].
         */
        ThisType subcurve(Scalar t0, Scalar t1) const {
            if (t0 > t1) {
                throw invalid_setting_error("t0 must be smaller than t1");
            }
            if (t0 < 0 || t0 > 1 || t1 < 0 || t1 > 1) {
                throw invalid_setting_error("Invalid range");
            }
            return split(t0)[1].split(t1)[0];
        }

};


}
