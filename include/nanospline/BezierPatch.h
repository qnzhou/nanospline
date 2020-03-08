#pragma once

#include <nanospline/PatchBase.h>
#include <nanospline/Bezier.h>

namespace nanospline {

template<typename _Scalar, int _dim=3,
    int _degree_u=3, int _degree_v=3>
class BezierPatch : PatchBase<_Scalar, _dim> {
    public:
        static_assert(_dim > 0, "Dimension must be positive.");
        static_assert(_degree_u > 0, "Degree in u must be positive.");
        static_assert(_degree_v > 0, "Degree in v must be positive.");

        using Base = PatchBase<_Scalar, _dim>;
        using Scalar = typename Base::Scalar;
        using Point = typename Base::Point;
        using ControlGrid = Eigen::Matrix<Scalar, (_degree_u+1)*(_degree_v+1), _dim>;
        using IsoCurveU = Bezier<Scalar, _dim, _degree_u>;
        using IsoCurveV = Bezier<Scalar, _dim, _degree_v>;

    public:
        Point evaluate(Scalar u, Scalar v) const override final {
            auto iso_curve_v = compute_iso_curve_v(u);
            return iso_curve_v.evaluate(v);
        }

        Point evaluate_derivative_u(Scalar u, Scalar v) const override final {
            auto iso_curve_u = compute_iso_curve_u(v);
            return iso_curve_u.evaluate_derivative(u);
        }

        Point evaluate_derivative_v(Scalar u, Scalar v) const override final {
            auto iso_curve_v = compute_iso_curve_v(u);
            return iso_curve_v.evaluate_derivative(v);
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

        IsoCurveU compute_iso_curve_u(Scalar v) const {
            typename IsoCurveU::ControlPoints control_points_u;
            for (int i=0; i<=_degree_u; i++) {
                typename IsoCurveV::ControlPoints control_points_v;
                for (int j=0; j<=_degree_v; j++) {
                    control_points_v.row(j) = get_control_point(i, j);
                }

                IsoCurveV iso_curve_v;
                iso_curve_v.set_control_points(std::move(control_points_v));
                control_points_u.row(i) = iso_curve_v.evaluate(v);
            }

            IsoCurveU iso_curve_u;
            iso_curve_u.set_control_points(std::move(control_points_u));
            return iso_curve_u;
        }

        IsoCurveV compute_iso_curve_v(Scalar u) const {
            typename IsoCurveV::ControlPoints control_points_v;
            for (int j=0; j<=_degree_v; j++) {
                typename IsoCurveU::ControlPoints control_points_u;
                for (int i=0; i<=_degree_u; i++) {
                    control_points_u.row(i) = get_control_point(i, j);
                }

                IsoCurveU iso_curve_u;
                iso_curve_u.set_control_points(std::move(control_points_u));
                control_points_v.row(j) = iso_curve_u.evaluate(u);
            }

            IsoCurveV iso_curve_v;
            iso_curve_v.set_control_points(std::move(control_points_v));
            return iso_curve_v;
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

    protected:
        Point get_control_point(int ui, int vj) const {
            return m_control_grid.row(vj*(_degree_u+1) + ui);
        }

    protected:
        ControlGrid m_control_grid;
};

}
