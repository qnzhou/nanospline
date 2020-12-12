#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <Eigen/src/Core/Matrix.h>
#include <nanospline/BezierBase.h>
#include <nanospline/Exceptions.h>

#if NANOSPLINE_SYMPY
#include <nanospline/internal/auto_inflection_Bezier.h>
#include <nanospline/internal/auto_match_tangent_Bezier.h>
#include <nanospline/internal/auto_singularity_Bezier.h>
#endif

namespace nanospline {

template <typename _Scalar, int _dim = 2, int _degree = 3, bool _generic = (_degree < 0)>
class Bezier final : public BezierBase<_Scalar, _dim, _degree, _generic>
{
public:
    using Base = BezierBase<_Scalar, _dim, _degree, _generic>;
    using ThisType = Bezier<_Scalar, _dim, _degree, _generic>;
    using Scalar = typename Base::Scalar;
    using Point = typename Base::Point;
    using ControlPoints = typename Base::ControlPoints;
    using BlossomVector = typename Base::BlossomVector;

public:
    Bezier() = default;

    std::unique_ptr<CurveBase<_Scalar, _dim>> clone() const override
    {
        return std::make_unique<ThisType>(*this);
    }

    Point evaluate(Scalar t) const override
    {
        int curve_degree = Base::get_degree();
        ControlPoints control_pts(Base::m_control_points);
        return deBoor(t, curve_degree, control_pts);
    }

    Scalar inverse_evaluate(const Point& p) const override
    {
        throw not_implemented_error("Too complex, sigh");
    }

    void get_derivative_coefficients(int curve_degree, ControlPoints& control_pts) const
    {
        int deriv_degree = curve_degree - 1;

        for (int i = 0; i < deriv_degree + 1; i++) {
            control_pts.row(i) = curve_degree * (control_pts.row(i + 1) - control_pts.row(i));
        }
    }

    Point evaluate_derivative(Scalar t) const override
    {
        const auto curve_degree = Base::get_degree();
        assert(curve_degree >= 0);

        if (curve_degree == 0) return Point::Zero();

        const int deriv_degree = curve_degree - 1;

        // Get control points defining the derivative
        ControlPoints control_pts(Base::m_control_points);
        get_derivative_coefficients(curve_degree, control_pts);

        // Evaluate the derivative curve
        return deBoor(t, deriv_degree, control_pts);
    }

    Point evaluate_2nd_derivative(Scalar t) const override
    {
        const auto curve_degree = Base::get_degree();
        assert(curve_degree >= 0);

        if (curve_degree <= 1) return Point::Zero();

        const int deriv_degree = curve_degree - 1;
        const int deriv2_degree = curve_degree - 2;

        // Get control points defining the second derivative
        ControlPoints control_pts(Base::m_control_points);
        get_derivative_coefficients(curve_degree, control_pts);
        get_derivative_coefficients(deriv_degree, control_pts);

        // Evaluate the second derivative curve
        return deBoor(t, deriv2_degree, control_pts);
    }

    std::vector<Scalar> compute_inflections(const Scalar lower, const Scalar upper) const override
    {
#if NANOSPLINE_SYMPY
        if (_dim != 2) {
            throw std::runtime_error("Inflection computation is for 2D curves only");
        }
        const auto degree = Base::get_degree();
        if (degree <= 2) {
            return {};
        }
        std::vector<Scalar> res;

        try {
            res = nanospline::internal::compute_Bezier_inflections(
                Base::m_control_points, lower, upper);
        } catch (infinite_root_error&) {
            // Infinitely many inflections.
            res.clear();
        }

        std::sort(res.begin(), res.end());
        res.erase(std::unique(res.begin(), res.end()), res.end());

        return res;
#else
        throw not_implemented_error("This feature require 'NANOSPLINE_SYMPY' compiler flag.");
        return {};
#endif
    }

    Scalar get_turning_angle(Scalar t0, Scalar t1) const override
    {
        if (_dim != 2) {
            throw invalid_setting_error("Turning angle is for 2D only");
        }
        auto segment = subcurve(t0, t1);
        const auto& ctrl_pts = segment.get_control_points();
        const auto num_ctrl_pts = ctrl_pts.rows();
        Scalar theta = 0;
        for (int i = 1; i + 1 < num_ctrl_pts; i++) {
            const Point v0 = ctrl_pts.row(i) - ctrl_pts.row(i - 1);
            const Point v1 = ctrl_pts.row(i + 1) - ctrl_pts.row(i);
            theta += std::atan2(v0[0] * v1[1] - v0[1] * v1[0], v0[0] * v1[0] + v0[1] * v1[1]);
        }
        return theta;
    }

    std::vector<Scalar> reduce_turning_angle(const Scalar lower, const Scalar upper) const override
    {
#if NANOSPLINE_SYMPY
        constexpr Scalar tol = static_cast<Scalar>(1e-8);
        if (_dim != 2) {
            throw std::runtime_error("Turning angle reduction is for 2D curves only");
        }
        const auto degree = Base::get_degree();
        if (degree < 2) {
            return {};
        }

        auto tan0 = evaluate_derivative(lower);
        auto tan1 = evaluate_derivative(upper);

        if (tan0.norm() < tol || tan1.norm() < tol) {
            std::vector<Scalar> res;
            res.push_back((lower + upper) / 2);
            return res;
        }

        tan0 = tan0 / tan0.norm();
        tan1 = tan1 / tan1.norm();

        const Eigen::Matrix<Scalar, 2, 1> ave_tangent(
            -(tan0[1] + tan1[1]) / 2, (tan0[0] + tan1[0]) / 2);

        std::vector<Scalar> res;
        try {
            res = nanospline::internal::match_tangent_bezier(
                Base::m_control_points, degree, ave_tangent, lower, upper);
        } catch (infinite_root_error&) {
            res.clear();
        }

        std::sort(res.begin(), res.end());
        res.erase(std::unique(res.begin(), res.end()), res.end());

        return res;
#else
        throw not_implemented_error("This feature require 'NANOSPLINE_SYMPY' compiler flag.");
        return {};
#endif
    }

    std::vector<Scalar> compute_singularities(
        const Scalar lower = 0.0, const Scalar upper = 1.0) const override
    {
#if NANOSPLINE_SYMPY
        if (_dim != 2) {
            throw std::runtime_error("Singularity computation is for 2D curves only");
        }

        std::vector<Scalar> res = nanospline::internal::compute_Bezier_singularities(
            Base::m_control_points, lower, upper);

        std::sort(res.begin(), res.end());
        res.erase(std::unique(res.begin(), res.end()), res.end());

        return res;
#else
        throw not_implemented_error("This feature require 'NANOSPLINE_SYMPY' compiler flag.");
        return {};
#endif
    }

    /**
     * Split a curve in halves at t.
     */
    std::vector<ThisType> split(Scalar t) const
    {
        if (!this->is_split_point_valid(t)) {
            return std::vector<ThisType>{*this};
        }

        // Copy intentionally.
        auto ctrl_pts = Base::get_control_points();
        const auto d = Base::get_degree();
        ControlPoints ctrl_pts_1(d + 1, _dim);
        ControlPoints ctrl_pts_2(d + 1, _dim);

        for (int i = 0; i <= d; i++) {
            ctrl_pts_1.row(i) = ctrl_pts.row(0);
            ctrl_pts_2.row(d - i) = ctrl_pts.row(d - i);
            for (int j = 0; j < d - i; j++) {
                ctrl_pts.row(j) = (1.0 - t) * ctrl_pts.row(j) + t * ctrl_pts.row(j + 1);
            }
        }

        std::vector<ThisType> results(2, ThisType());
        results[0].set_control_points(std::move(ctrl_pts_1));
        results[1].set_control_points(std::move(ctrl_pts_2));
        return results;
    }

    std::vector<ThisType> _split(Scalar t) const
    {
        assert(t >= Scalar(0));
        assert(t <= Scalar(1));

        std::vector<ThisType> results(2, ThisType());
        results[0] = subcurve(Scalar(0.), t);
        results[1] = subcurve(t, Scalar(1.));
        return results;
    }

    /**
     * Extract a subcurve in range [t0, t1].
     */
    ThisType subcurve(Scalar t0, Scalar t1) const
    {
        if (t0 > t1) {
            throw invalid_setting_error("t0 must be smaller than t1");
        }
        if (t0 < 0 || t0 > 1 || t1 < 0 || t1 > 1) {
            throw invalid_setting_error("Invalid range");
        }
        // To get the coefficients of a subcurve defined on a subdomain
        // [t0,t1], each coefficient is given by a blossom:
        //    c_i = b[t0, ..., t0, t1, ..., t1]
        //             |________|   |________|
        //              =degree-i       =i
        // for i = 0, ... ,degree

        const auto curve_degree = Base::get_degree();
        ControlPoints subcurve_control_pts(curve_degree + 1, _dim);

        for (int i = 0; i < curve_degree + 1; i++) {
            // Form the blossom vector [t0, ..., t0, t1, ..., t1]
            BlossomVector blossom_vector(curve_degree);
            blossom_vector.setConstant(t1);

            for (int j = 0; j < i; j++) {
                blossom_vector(j) = t0;
            }
            // Evaluate the blossom
            ControlPoints control_points(Base::m_control_points);
            blossom(blossom_vector, curve_degree, control_points);

            subcurve_control_pts.row(curve_degree - i) = control_points.row(curve_degree);
        }

        ThisType subcurve;
        subcurve.set_control_points(subcurve_control_pts);

        return subcurve;
    }

    /**
     * Return the same curve but with degree increased by 1.
     */
    Bezier < _Scalar, _dim, _degree<0 ? _degree : _degree + 1, _generic> elevate_degree() const
    {
        using TargetType = Bezier < _Scalar, _dim, _degree<0 ? _degree : _degree + 1, _generic>;
        const auto degree = Base::get_degree();

        const auto& ctrl_pts = Base::m_control_points;
        typename TargetType::ControlPoints target_ctrl_pts(degree + 2, _dim);
        assert(ctrl_pts.rows() == degree + 1);

        target_ctrl_pts.row(0) = ctrl_pts.row(0);
        for (int i = 1; i < degree + 1; i++) {
            const Scalar alpha = (Scalar)i / (Scalar)(degree + 1);
            target_ctrl_pts.row(i) = (1 - alpha) * ctrl_pts.row(i) + alpha * ctrl_pts.row(i - 1);
        }
        target_ctrl_pts.row(degree + 1) = ctrl_pts.row(degree);

        TargetType new_curve;
        new_curve.set_control_points(std::move(target_ctrl_pts));
        return new_curve;
    }

    static Eigen::MatrixXd form_least_squares_matrix(Eigen::MatrixXd parameters)
    {
        const int num_control_pts = _degree + 1;

        Eigen::MatrixXd basis_func_control_pts(num_control_pts, 1);
        Bezier<Scalar, 1, _degree, false> basis_function;
        basis_func_control_pts.setZero();

        // Suppose we are fitting samples p_0, ..., p_n of a function f, with
        // parameter values t_0, ..., t_n. If B_j^n(t) is the jth basis function
        // then least_squares_matrix(i,j) = B_j^n(t_i)
        const int num_constraints = int(parameters.rows());
        Eigen::MatrixXd least_squares_matrix =
            Eigen::MatrixXd::Zero(num_constraints, num_control_pts);

        for (int j = 0; j < num_control_pts; j++) {
            basis_func_control_pts.row(j) << 1.;
            basis_function.set_control_points(basis_func_control_pts);
            for (int i = 0; i < num_constraints; i++) {
                Scalar t = parameters(i, 0);
                Scalar bezier_value = basis_function.evaluate(t)(0);
                least_squares_matrix(i, j) = bezier_value;
            }
            basis_func_control_pts.row(j) << 0.;
        }
        return least_squares_matrix;
    }

    static ThisType fit(Eigen::MatrixXd parameters, Eigen::MatrixXd values)
    {
        ThisType least_squares_fit;
        assert(parameters.rows() == values.rows());
        assert(parameters.cols() == 1);
        assert(values.cols() == least_squares_fit.get_dim());
        const int num_control_pts = _degree + 1;

        // 1. Form least squares matrix .
        Eigen::MatrixXd least_squares_matrix = form_least_squares_matrix(parameters);

        // 2. Least squares solve via SVD
        Eigen::Matrix<Scalar, num_control_pts, _dim> fit_control_points =
            least_squares_matrix.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(values);

        // 3. Store control points in least_squares_fit and return
        least_squares_fit.set_control_points(fit_control_points);
        return least_squares_fit;
    }

    void deform(Eigen::MatrixXd parameters, Eigen::MatrixXd changes_in_values)
    {
        ThisType least_squares_fit = ThisType::fit(parameters, changes_in_values);

        ControlPoints changes_in_control_points = least_squares_fit.get_control_points();
        ControlPoints updated_control_points = Base::m_control_points + changes_in_control_points;
        Base::set_control_points(updated_control_points);
    }

private:
    void blossom(const BlossomVector& blossom_vector, int degree, ControlPoints& control_pts) const
    {
        // Unfurl the standard de Boor recursion into two loops:
        // if degree == 0
        //   return c_i
        // else:
        //   return t*c_i + (1-t)*c_{i-1}
        // The outer loop corresponds to the recursion; we build the final value
        // in a bottom-up fashion. The base case degree==0 is implicitly
        // performed because the control points are given as input.
        // We then interpolate between (degree - rec_step) control points.
        // After (degree) interpolations, the final value lives in
        // control_pts(degree).
        for (int rec_step = 1; rec_step <= degree; rec_step++) {
            for (int j = degree; j >= rec_step; j--) {
                Scalar t = blossom_vector(rec_step - 1);
                control_pts.row(j) = t * control_pts.row(j) + (1. - t) * control_pts.row(j - 1);
            }
        }
    }

    Point deBoor(Scalar t, int degree, ControlPoints& control_pts) const
    {
        // This function returns the degree+1 control points of a Bezier curve
        // of degree "degree" after applying "degree" iterations of deBoor's
        // algorithm
        if (degree > 0) {
            // Set t_i=t for all blossom evaluation points
            BlossomVector blossom_vector(Base::get_degree());
            blossom_vector.setConstant(t);

            // Evaluate the blossom
            blossom(blossom_vector, degree, control_pts);
        }
        return control_pts.row(degree);
    }
};

template <typename _Scalar, int _dim>
class Bezier<_Scalar, _dim, 0, false> final : public BezierBase<_Scalar, _dim, 0, false>
{
public:
    using Base = BezierBase<_Scalar, _dim, 0, false>;
    using Scalar = typename Base::Scalar;
    using Point = typename Base::Point;
    using ControlPoints = typename Base::ControlPoints;

public:
    std::unique_ptr<CurveBase<_Scalar, _dim>> clone() const override
    {
        return std::make_unique<Bezier<_Scalar, _dim, 0, false>>(*this);
    }

    Point evaluate(Scalar t) const override { return Base::m_control_points; }

    Scalar inverse_evaluate(const Point& p) const override { return 0.0; }

    Point evaluate_derivative(Scalar t) const override { return Point::Zero(); }

    Point evaluate_2nd_derivative(Scalar t) const override { return Point::Zero(); }

    std::vector<Scalar> compute_inflections(
        const Scalar lower = 0.0, const Scalar upper = 1.0) const override
    {
        return {};
    }

    Scalar get_turning_angle(Scalar t0, Scalar t1) const override { return 0; }

    std::vector<Scalar> reduce_turning_angle(const Scalar lower, const Scalar upper) const override
    {
        return {};
    }

    std::vector<Scalar> compute_singularities(const Scalar lower, const Scalar upper) const override
    {
        return {};
    }

public:
    Bezier<_Scalar, _dim, 1, false> elevate_degree() const
    {
        using TargetType = Bezier<_Scalar, _dim, 1, false>;
        TargetType new_curve;
        typename TargetType::ControlPoints ctrl_pts(2, _dim);
        ctrl_pts.row(0) = Base::m_control_points.row(0);
        ctrl_pts.row(1) = Base::m_control_points.row(0);
        new_curve.set_control_points(std::move(ctrl_pts));
        return new_curve;
    }
};

template <typename _Scalar, int _dim>
class Bezier<_Scalar, _dim, 1, false> final : public BezierBase<_Scalar, _dim, 1, false>
{
public:
    using Base = BezierBase<_Scalar, _dim, 1, false>;
    using ThisType = Bezier<_Scalar, _dim, 1, false>;
    using Scalar = typename Base::Scalar;
    using Point = typename Base::Point;
    using ControlPoints = typename Base::ControlPoints;

public:
    std::unique_ptr<CurveBase<_Scalar, _dim>> clone() const override
    {
        return std::make_unique<Bezier<_Scalar, _dim, 1, false>>(*this);
    }

    Point evaluate(Scalar t) const override
    {
        return (1.0 - t) * Base::m_control_points.row(0) + t * Base::m_control_points.row(1);
    }

    Scalar inverse_evaluate(const Point& p) const override
    {
        Point e = Base::m_control_points.row(1) - Base::m_control_points.row(0);
        Scalar t = (p - Base::m_control_points.row(0)).dot(e) / e.squaredNorm();
        return std::max<Scalar>(std::min<Scalar>(t, 1.0), 0.0);
    }

    Point evaluate_derivative(Scalar t) const override
    {
        return Base::m_control_points.row(1) - Base::m_control_points.row(0);
    }

    Point evaluate_2nd_derivative(Scalar t) const override { return Point::Zero(); }

    std::vector<Scalar> compute_inflections(
        const Scalar lower = 0.0, const Scalar upper = 1.0) const override
    {
        return {};
    }

    Scalar get_turning_angle(Scalar t0, Scalar t1) const override { return 0; }

    std::vector<Scalar> reduce_turning_angle(const Scalar lower, const Scalar upper) const override
    {
        return {};
    }

    std::vector<Scalar> compute_singularities(const Scalar lower, const Scalar upper) const override
    {
        return {};
    }

public:
    Bezier<_Scalar, _dim, 2, false> elevate_degree() const
    {
        using TargetType = Bezier<_Scalar, _dim, 2, false>;
        TargetType new_curve;
        typename TargetType::ControlPoints ctrl_pts(3, _dim);
        ctrl_pts.row(0) = Base::m_control_points.row(0);
        ctrl_pts.row(1) = Base::m_control_points.colwise().mean();
        ctrl_pts.row(2) = Base::m_control_points.row(1);
        new_curve.set_control_points(std::move(ctrl_pts));
        return new_curve;
    }

    std::vector<ThisType> split(Scalar t) const
    {
        if (!this->is_split_point_valid(t)) {
            return std::vector<ThisType>{*this};
        }
        // Copy intentionally.
        auto ctrl_pts = Base::get_control_points();

        ControlPoints ctrl_pts_1(2, _dim);
        ControlPoints ctrl_pts_2(2, _dim);

        for (int i = 0; i <= 1; i++) {
            ctrl_pts_1.row(i) = ctrl_pts.row(0);
            ctrl_pts_2.row(1 - i) = ctrl_pts.row(1 - i);
            for (int j = 0; j < 1 - i; j++) {
                ctrl_pts.row(j) = (1.0 - t) * ctrl_pts.row(j) + t * ctrl_pts.row(j + 1);
            }
        }

        std::vector<ThisType> results(2, ThisType());
        results[0].set_control_points(std::move(ctrl_pts_1));
        results[1].set_control_points(std::move(ctrl_pts_2));

        return results;
    }
};

template <typename _Scalar, int _dim>
class Bezier<_Scalar, _dim, 2, false> final : public BezierBase<_Scalar, _dim, 2, false>
{
public:
    using Base = BezierBase<_Scalar, _dim, 2, false>;
    using ThisType = Bezier<_Scalar, _dim, 2, false>;
    using Scalar = typename Base::Scalar;
    using Point = typename Base::Point;
    using ControlPoints = typename Base::ControlPoints;

public:
    std::unique_ptr<CurveBase<_Scalar, _dim>> clone() const override
    {
        return std::make_unique<Bezier<_Scalar, _dim, 2, false>>(*this);
    }

    Point evaluate(Scalar t) const override
    {
        const Point p0 =
            (1.0 - t) * Base::m_control_points.row(0) + t * Base::m_control_points.row(1);
        const Point p1 =
            (1.0 - t) * Base::m_control_points.row(1) + t * Base::m_control_points.row(2);
        return (1.0 - t) * p0 + t * p1;
    }

    Scalar inverse_evaluate(const Point& p) const override
    {
        throw not_implemented_error("Too complex, sigh");
    }

    Point evaluate_derivative(Scalar t) const override
    {
        const Point p0 =
            (1.0 - t) * Base::m_control_points.row(0) + t * Base::m_control_points.row(1);
        const Point p1 =
            (1.0 - t) * Base::m_control_points.row(1) + t * Base::m_control_points.row(2);
        return (p1 - p0) * 2;
    }

    Point evaluate_2nd_derivative(Scalar t) const override
    {
        const auto& ctrl_pts = Base::m_control_points;
        return 2 * (ctrl_pts.row(0) + ctrl_pts.row(2) - 2 * ctrl_pts.row(1));
    }

    std::vector<Scalar> compute_inflections(
        const Scalar lower = 0.0, const Scalar upper = 1.0) const override
    {
        return {};
    }

    Scalar get_turning_angle(Scalar t0, Scalar t1) const override
    {
        if (_dim != 2) {
            throw invalid_setting_error("Turning angle is for 2D only");
        }

        const Point v0 = evaluate_derivative(t0);
        const Point v1 = evaluate_derivative(t1);
        return std::atan2(v0[0] * v1[1] - v0[1] * v1[0], v0[0] * v1[0] + v0[1] * v1[1]);
    }

    std::vector<Scalar> reduce_turning_angle(const Scalar lower, const Scalar upper) const override
    {
#if NANOSPLINE_SYMPY
        constexpr Scalar tol = static_cast<Scalar>(1e-8);
        if (_dim != 2) {
            throw std::runtime_error("Turning angle reduction is for 2D curves only");
        }

        auto tan0 = evaluate_derivative(lower);
        auto tan1 = evaluate_derivative(upper);

        if (tan0.norm() < tol || tan1.norm() < tol) {
            std::vector<Scalar> res;
            res.push_back((lower + upper) / 2);
            return res;
        }

        tan0 = tan0 / tan0.norm();
        tan1 = tan1 / tan1.norm();

        const Eigen::Matrix<Scalar, 2, 1> ave_tangent(
            -(tan0[1] + tan1[1]) / 2, (tan0[0] + tan1[0]) / 2);

        std::vector<Scalar> res;
        try {
            res = nanospline::internal::match_tangent_Bezier_degree_2(Base::m_control_points(0, 0),
                Base::m_control_points(0, 1),
                Base::m_control_points(1, 0),
                Base::m_control_points(1, 1),
                Base::m_control_points(2, 0),
                Base::m_control_points(2, 1),
                ave_tangent,
                lower,
                upper);
        } catch (infinite_root_error&) {
            res.clear();
        }

        std::sort(res.begin(), res.end());
        res.erase(std::unique(res.begin(), res.end()), res.end());

        return res;
#else
        throw not_implemented_error("This feature require 'NANOSPLINE_SYMPY' compiler flag.");
        return {};
#endif
    }

    std::vector<Scalar> compute_singularities(const Scalar lower, const Scalar upper) const override
    {
#if NANOSPLINE_SYMPY
        if (_dim != 2) {
            throw std::runtime_error("Singularity computation is for 2D curves only");
        }

        auto res = nanospline::internal::compute_Bezier_degree_2_singularities(
            Base::m_control_points(0, 0),
            Base::m_control_points(0, 1),
            Base::m_control_points(1, 0),
            Base::m_control_points(1, 1),
            Base::m_control_points(2, 0),
            Base::m_control_points(2, 1),
            lower,
            upper);

        std::sort(res.begin(), res.end());
        res.erase(std::unique(res.begin(), res.end()), res.end());

        return res;
#else
        throw not_implemented_error("This feature require 'NANOSPLINE_SYMPY' compiler flag.");
        return {};
#endif
    }

public:
    Bezier<_Scalar, _dim, 3, false> elevate_degree() const
    {
        using TargetType = Bezier<_Scalar, _dim, 3, false>;
        TargetType new_curve;
        typename TargetType::ControlPoints ctrl_pts(4, _dim);
        ctrl_pts.row(0) = Base::m_control_points.row(0);
        ctrl_pts.row(1) =
            Base::m_control_points.row(0) / 3.0 + Base::m_control_points.row(1) * 2.0 / 3.0;
        ctrl_pts.row(2) =
            Base::m_control_points.row(1) * 2.0 / 3.0 + Base::m_control_points.row(2) / 3.0;
        ctrl_pts.row(3) = Base::m_control_points.row(2);
        new_curve.set_control_points(std::move(ctrl_pts));
        return new_curve;
    }

    std::vector<ThisType> split(Scalar t) const
    {
        if (!this->is_split_point_valid(t)) {
            return std::vector<ThisType>{*this};
        }
        // Copy intentionally.
        auto ctrl_pts = Base::get_control_points();

        ControlPoints ctrl_pts_1(3, _dim);
        ControlPoints ctrl_pts_2(3, _dim);

        for (int i = 0; i <= 2; i++) {
            ctrl_pts_1.row(i) = ctrl_pts.row(0);
            ctrl_pts_2.row(2 - i) = ctrl_pts.row(2 - i);
            for (int j = 0; j < 2 - i; j++) {
                ctrl_pts.row(j) = (1.0 - t) * ctrl_pts.row(j) + t * ctrl_pts.row(j + 1);
            }
        }

        std::vector<ThisType> results(2, ThisType());
        results[0].set_control_points(std::move(ctrl_pts_1));
        results[1].set_control_points(std::move(ctrl_pts_2));

        return results;
    }
};

template <typename _Scalar, int _dim>
class Bezier<_Scalar, _dim, 3, false> final : public BezierBase<_Scalar, _dim, 3, false>
{
public:
    using Base = BezierBase<_Scalar, _dim, 3, false>;
    using ThisType = Bezier<_Scalar, _dim, 3, false>;
    using Scalar = typename Base::Scalar;
    using Point = typename Base::Point;
    using ControlPoints = typename Base::ControlPoints;

public:
    std::unique_ptr<CurveBase<_Scalar, _dim>> clone() const override
    {
        return std::make_unique<Bezier<_Scalar, _dim, 3, false>>(*this);
    }

    Point evaluate(Scalar t) const override
    {
        const Point q0 =
            (1.0 - t) * Base::m_control_points.row(0) + t * Base::m_control_points.row(1);
        const Point q1 =
            (1.0 - t) * Base::m_control_points.row(1) + t * Base::m_control_points.row(2);
        const Point q2 =
            (1.0 - t) * Base::m_control_points.row(2) + t * Base::m_control_points.row(3);

        const Point p0 = (1.0 - t) * q0 + t * q1;
        const Point p1 = (1.0 - t) * q1 + t * q2;
        return (1.0 - t) * p0 + t * p1;
    }

    Scalar inverse_evaluate(const Point& p) const override
    {
        throw not_implemented_error("Too complex, sigh");
    }

    Point evaluate_derivative(Scalar t) const override
    {
        const Point q0 =
            (1.0 - t) * Base::m_control_points.row(0) + t * Base::m_control_points.row(1);
        const Point q1 =
            (1.0 - t) * Base::m_control_points.row(1) + t * Base::m_control_points.row(2);
        const Point q2 =
            (1.0 - t) * Base::m_control_points.row(2) + t * Base::m_control_points.row(3);

        const Point p0 = (1.0 - t) * q0 + t * q1;
        const Point p1 = (1.0 - t) * q1 + t * q2;

        return (p1 - p0) * 3;
    }

    Point evaluate_2nd_derivative(Scalar t) const override
    {
        const Point q0 =
            (1.0 - t) * Base::m_control_points.row(0) + t * Base::m_control_points.row(1);
        const Point q1 =
            (1.0 - t) * Base::m_control_points.row(1) + t * Base::m_control_points.row(2);
        const Point q2 =
            (1.0 - t) * Base::m_control_points.row(2) + t * Base::m_control_points.row(3);
        return 6 * (q0 + q2 - 2 * q1);
    }

    std::vector<Scalar> compute_inflections(const Scalar lower, const Scalar upper) const override
    {
#if NANOSPLINE_SYMPY
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
                lower,
                upper);
        } catch (infinite_root_error&) {
            res.clear();
        }

        std::sort(res.begin(), res.end());
        res.erase(std::unique(res.begin(), res.end()), res.end());

        return res;
#else
        throw not_implemented_error("This feature require 'NANOSPLINE_SYMPY' compiler flag.");
        return {};
#endif
    }

    Scalar get_turning_angle(Scalar t0, Scalar t1) const override
    {
        if (_dim != 2) {
            throw invalid_setting_error("Turning angle is for 2D only");
        }

        const auto piece = subcurve(t0, t1);
        const auto& ctrl_pts = piece.get_control_points();

        const Point v0 = ctrl_pts.row(1) - ctrl_pts.row(0);
        const Point v1 = ctrl_pts.row(2) - ctrl_pts.row(1);
        const Point v2 = ctrl_pts.row(3) - ctrl_pts.row(2);

        return std::atan2(v0[0] * v1[1] - v0[1] * v1[0], v0[0] * v1[0] + v0[1] * v1[1]) +
               std::atan2(v1[0] * v2[1] - v1[1] * v2[0], v1[0] * v2[0] + v1[1] * v2[1]);
    }

    std::vector<Scalar> reduce_turning_angle(const Scalar lower, const Scalar upper) const override
    {
#if NANOSPLINE_SYMPY
        constexpr Scalar tol = static_cast<Scalar>(1e-8);
        if (_dim != 2) {
            throw std::runtime_error("Turning angle reduction is for 2D curves only");
        }

        auto tan0 = evaluate_derivative(lower);
        auto tan1 = evaluate_derivative(upper);

        if (tan0.norm() < tol || tan1.norm() < tol || (tan0 + tan1).norm() < 2 * tol) {
            std::vector<Scalar> res;
            res.push_back((lower + upper) / 2);
            return res;
        }

        tan0 = tan0 / tan0.norm();
        tan1 = tan1 / tan1.norm();

        const Eigen::Matrix<Scalar, 2, 1> ave_tangent(
            -(tan0[1] + tan1[1]) / 2, (tan0[0] + tan1[0]) / 2);

        std::vector<Scalar> res;
        try {
            res = nanospline::internal::match_tangent_Bezier_degree_3(Base::m_control_points(0, 0),
                Base::m_control_points(0, 1),
                Base::m_control_points(1, 0),
                Base::m_control_points(1, 1),
                Base::m_control_points(2, 0),
                Base::m_control_points(2, 1),
                Base::m_control_points(3, 0),
                Base::m_control_points(3, 1),
                ave_tangent,
                lower,
                upper);
        } catch (infinite_root_error&) {
            res.clear();
        }

        std::sort(res.begin(), res.end());
        res.erase(std::unique(res.begin(), res.end()), res.end());

        return res;
#else
        throw not_implemented_error("This feature require 'NANOSPLINE_SYMPY' compiler flag.");
        return {};
#endif
    }

    std::vector<Scalar> compute_singularities(const Scalar lower, const Scalar upper) const override
    {
#if NANOSPLINE_SYMPY
        if (_dim != 2) {
            throw std::runtime_error("Singularity computation is for 2D curves only");
        }

        std::vector<Scalar> res = nanospline::internal::compute_Bezier_degree_3_singularities(
            Base::m_control_points(0, 0),
            Base::m_control_points(0, 1),
            Base::m_control_points(1, 0),
            Base::m_control_points(1, 1),
            Base::m_control_points(2, 0),
            Base::m_control_points(2, 1),
            Base::m_control_points(3, 0),
            Base::m_control_points(3, 1),
            lower,
            upper);

        std::sort(res.begin(), res.end());
        res.erase(std::unique(res.begin(), res.end()), res.end());

        return res;
#else
        throw not_implemented_error("This feature require 'NANOSPLINE_SYMPY' compiler flag.");
        return {};
#endif
    }

public:
    std::vector<ThisType> split(Scalar t) const
    {
        if (!this->is_split_point_valid(t)) {
            return std::vector<ThisType>{*this};
        }
        // Copy intentionally.
        auto ctrl_pts = Base::get_control_points();

        ControlPoints ctrl_pts_1(4, _dim);
        ControlPoints ctrl_pts_2(4, _dim);

        for (int i = 0; i <= 3; i++) {
            ctrl_pts_1.row(i) = ctrl_pts.row(0);
            ctrl_pts_2.row(3 - i) = ctrl_pts.row(3 - i);
            for (int j = 0; j < 3 - i; j++) {
                ctrl_pts.row(j) = (1.0 - t) * ctrl_pts.row(j) + t * ctrl_pts.row(j + 1);
            }
        }

        std::vector<ThisType> results(2, ThisType());
        results[0].set_control_points(std::move(ctrl_pts_1));
        results[1].set_control_points(std::move(ctrl_pts_2));

        return results;
    }

    /**
     * Extract a subcurve in range [t0, t1].
     */
    ThisType subcurve(Scalar t0, Scalar t1) const
    {
        if (t0 > t1) {
            throw invalid_setting_error("t0 must be smaller than t1");
        }
        if (this->is_split_point_valid(t0) && this->is_split_point_valid(t1)) {
        }
        bool t0_is_endpt = t0 == this->get_domain_lower_bound();
        bool t1_is_endpt = t1 == this->get_domain_upper_bound();

        if (t0_is_endpt && t1_is_endpt) {
            return *this; // identity: return the current curve
        } else if (t0_is_endpt) {
            return split(t1)[0]; // curve from [0, t1]
        } else if (t1_is_endpt) {
            return split(t0)[1]; // curve from [t0, 1]
        } else {
            return split(t0)[1].split(t1)[0];
        }
    }

    Bezier<_Scalar, _dim, 4, false> elevate_degree() const
    {
        using TargetType = Bezier<_Scalar, _dim, 4, false>;
        TargetType new_curve;
        typename TargetType::ControlPoints ctrl_pts(5, _dim);
        ctrl_pts.row(0) = Base::m_control_points.row(0);
        ctrl_pts.row(1) =
            Base::m_control_points.row(0) / 4.0 + Base::m_control_points.row(1) * 3.0 / 4.0;
        ctrl_pts.row(2) = Base::m_control_points.row(1) / 2.0 + Base::m_control_points.row(2) / 2.0;
        ctrl_pts.row(3) =
            Base::m_control_points.row(2) * 3.0 / 4.0 + Base::m_control_points.row(3) / 4.0;
        ctrl_pts.row(4) = Base::m_control_points.row(3);
        new_curve.set_control_points(std::move(ctrl_pts));
        return new_curve;
    }
};

} // namespace nanospline
