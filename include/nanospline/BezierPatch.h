#pragma once

#include <nanospline/PatchBase.h>
#include <nanospline/Bezier.h>
#include <nanospline/Exceptions.h>

namespace nanospline {

template<typename _Scalar, int _dim=3,
    int _degree_u=3, int _degree_v=3>
class BezierPatch final : public PatchBase<_Scalar, _dim> {
    public:
        static_assert(_dim > 0, "Dimension must be positive.");

        using Base = PatchBase<_Scalar, _dim>;
        using Scalar = typename Base::Scalar;
        using Point = typename Base::Point;
        using ControlGrid = typename Base::ControlGrid;
        using IsoCurveU = Bezier<Scalar, _dim, _degree_u>;
        using IsoCurveV = Bezier<Scalar, _dim, _degree_v>;

    public:
        BezierPatch() {
            Base::set_degree_u(_degree_u);
            Base::set_degree_v(_degree_v);
        }

    public:
        Point evaluate(Scalar u, Scalar v) const override {
            auto iso_curve_v = compute_iso_curve_v(u);
            return iso_curve_v.evaluate(v);
        }

        Point evaluate_derivative_u(Scalar u, Scalar v) const override {
            auto iso_curve_u = compute_iso_curve_u(v);
            return iso_curve_u.evaluate_derivative(u);
        }

        Point evaluate_derivative_v(Scalar u, Scalar v) const override {
            auto iso_curve_v = compute_iso_curve_v(u);
            return iso_curve_v.evaluate_derivative(v);
        }

        void initialize() override {
            const int degree_u = Base::get_degree_u();
            const int degree_v = Base::get_degree_v();
            if (Base::m_control_grid.rows() != (degree_u+1) * (degree_v+1)) {
                throw invalid_setting_error(
                        "Control grid size mismatch uv degrees");
            }
        }

    public:
        IsoCurveU compute_iso_curve_u(Scalar v) const {
            typename IsoCurveU::ControlPoints control_points_u;
            const int degree_u = Base::get_degree_u();
            const int degree_v = Base::get_degree_v();
            for (int i=0; i<=degree_u; i++) {
                typename IsoCurveV::ControlPoints control_points_v;
                for (int j=0; j<=degree_v; j++) {
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
            const int degree_u = Base::get_degree_u();
            const int degree_v = Base::get_degree_v();
            for (int j=0; j<=degree_v; j++) {
                typename IsoCurveU::ControlPoints control_points_u;
                for (int i=0; i<=degree_u; i++) {
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
            const int degree_u = Base::get_degree_u();
            return Base::m_control_grid.row(vj*(degree_u+1) + ui);
        }
};

}
