#pragma once

#include <nanospline/PatchBase.h>
#include <nanospline/RationalBezier.h>
#include <nanospline/BezierPatch.h>

namespace nanospline {

template<typename _Scalar, int _dim=3,
    int _degree_u=3, int _degree_v=3>
class RationalBezierPatch final : PatchBase<_Scalar, _dim> {
    public:
        static_assert(_dim > 0, "Dimension must be positive.");
        static_assert(_degree_u > 0, "Degree in u must be positive.");
        static_assert(_degree_v > 0, "Degree in v must be positive.");

        using Base = PatchBase<_Scalar, _dim>;
        using Scalar = typename Base::Scalar;
        using Point = typename Base::Point;
        using ControlGrid = Eigen::Matrix<Scalar, (_degree_u+1)*(_degree_v+1), _dim>;
        using Weights = Eigen::Matrix<Scalar, (_degree_u+1)*(_degree_v+1), 1>;
        using BezierPatchHomogeneous = BezierPatch<Scalar, _dim+1, _degree_u, _degree_v>;
        using IsoCurveU = RationalBezier<Scalar, _dim, _degree_u>;
        using IsoCurveV = RationalBezier<Scalar, _dim, _degree_v>;

    public:
        Point evaluate(Scalar u, Scalar v) const override {
            validate_initialization();
            const auto p = m_homogeneous.evaluate(u, v);
            return p.template segment<_dim>(0) / p[_dim];
        }

        Point evaluate_derivative_u(Scalar u, Scalar v) const override {
            validate_initialization();
            const auto p = m_homogeneous.evaluate(u, v);
            const auto d =
                m_homogeneous.evaluate_derivative_u(u, v);

            return (d.template head<_dim>() -
                    p.template head<_dim>() * d[_dim] / p[_dim])
                / p[_dim];
        }

        Point evaluate_derivative_v(Scalar u, Scalar v) const override {
            validate_initialization();
            const auto p = m_homogeneous.evaluate(u, v);
            const auto d =
                m_homogeneous.evaluate_derivative_v(u, v);

            return (d.template head<_dim>() -
                    p.template head<_dim>() * d[_dim] / p[_dim])
                / p[_dim];
        }

    public:
        constexpr int get_degree_u() const {
            return _degree_u;
        }

        constexpr int get_degree_v() const {
            return _degree_v;
        }

        const ControlGrid& get_control_grid() const {
            return m_control_grid;
        }

        template<typename Derived>
        void set_control_grid(const Eigen::PlainObjectBase<Derived>& ctrl_grid) {
            m_control_grid = ctrl_grid;
        }

        template<typename Derived>
        void set_control_grid(Eigen::PlainObjectBase<Derived>&& ctrl_grid) {
            m_control_grid.swap(ctrl_grid);
        }

        const Weights get_weights() const {
            return m_weights;
        }

        template<typename Derived>
        void set_weights(const Eigen::PlainObjectBase<Derived>& weights) {
            m_weights = weights;
        }

        template<typename Derived>
        void set_weights(Eigen::PlainObjectBase<Derived>&& weights) {
            m_weights.swap(weights);
        }

        void initialize() {
            typename BezierPatchHomogeneous::ControlGrid ctrl_pts(
                    m_control_grid.rows(), _dim+1);
            ctrl_pts.template leftCols<_dim>() =
                m_control_grid.array().colwise() * m_weights.array();
            ctrl_pts.template rightCols<1>() = m_weights;

            m_homogeneous.set_control_grid(std::move(ctrl_pts));
        }

        Scalar get_u_lower_bound() const {
            return 0.0;
        }

        Scalar get_v_lower_bound() const {
            return 0.0;
        }

        Scalar get_u_upper_bound() const {
            return 1.0;
        }

        Scalar get_v_upper_bound() const {
            return 1.0;
        }

        IsoCurveU compute_iso_curve_u(Scalar v) const {
            auto curve = m_homogeneous.compute_iso_curve_u(v);
            IsoCurveU iso_curve;
            iso_curve.set_homogeneous(curve);
            return iso_curve;
        }

        IsoCurveV compute_iso_curve_v(Scalar u) const {
            auto curve = m_homogeneous.compute_iso_curve_v(u);
            IsoCurveV iso_curve;
            iso_curve.set_homogeneous(curve);
            return iso_curve;
        }

        const BezierPatchHomogeneous& get_homogeneous() const {
            return m_homogeneous;
        }

    protected:
        std::tuple<Point, Scalar> get_control_point_and_weight(int ui, int vj) const {
            int index = vj*(_degree_u+1) + ui;
            return std::make_tuple(m_control_grid.row(index), m_weights[index]);
        }

        void validate_initialization() const {
            const auto& ctrl_pts = m_homogeneous.get_control_grid();
            if (ctrl_pts.rows() != m_control_grid.rows() ||
                ctrl_pts.rows() != m_weights.rows() ) {
                throw invalid_setting_error("Rational Bezier patch is not initialized.");
            }
        }

    protected:
        ControlGrid m_control_grid;
        Weights m_weights;
        BezierPatchHomogeneous m_homogeneous;
};

}
