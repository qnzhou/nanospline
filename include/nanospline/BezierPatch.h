#pragma once

#include <nanospline/PatchBase.h>
#include <nanospline/Bezier.h>
#include <nanospline/Exceptions.h>

namespace nanospline {

template<typename _Scalar, int _dim=3,
    int _degree_u=3, int _degree_v=3>
class BezierPatch final : public PatchBase<_Scalar, _dim> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        static_assert(_dim > 0, "Dimension must be positive.");

        using Base = PatchBase<_Scalar, _dim>;
        using Scalar = typename Base::Scalar;
        using Point = typename Base::Point;
        using ControlGrid = typename Base::ControlGrid;
        using IsoCurveU = Bezier<Scalar, _dim, _degree_u>;
        using IsoCurveV = Bezier<Scalar, _dim, _degree_v>;

    public:
        static BezierPatch<_Scalar, _dim, _degree_u, _degree_v> ZeroPatch() {
            BezierPatch<_Scalar, _dim, _degree_u, _degree_v> patch;
            patch.set_degree_u(_degree_u>0?_degree_u:0);
            patch.set_degree_v(_degree_v>0?_degree_v:0);
            ControlGrid grid((patch.get_degree_u()+1) * (patch.get_degree_v()+1), _dim);
            grid.setZero();
            patch.swap_control_grid(grid);
            patch.initialize();
            return patch;
        }

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

        Point evaluate_2nd_derivative_uu(Scalar u, Scalar v) const override {
            auto iso_curve_u = compute_iso_curve_u(v);
            return iso_curve_u.evaluate_2nd_derivative(u);
        }

        Point evaluate_2nd_derivative_vv(Scalar u, Scalar v) const override {
            auto iso_curve_v = compute_iso_curve_v(u);
            return iso_curve_v.evaluate_2nd_derivative(v);
        }

        Point evaluate_2nd_derivative_uv(Scalar u, Scalar v) const override {
            auto duv_patch = compute_duv_patch();
            return duv_patch.evaluate(u, v);
        }

        void initialize() override {
            const int degree_u = Base::get_degree_u();
            const int degree_v = Base::get_degree_v();
            if (Base::m_control_grid.rows() != (degree_u+1) * (degree_v+1)) {
                throw invalid_setting_error(
                        "Control grid size mismatch uv degrees");
            }
        }

        Scalar get_u_lower_bound() const override {
            return 0.0;
        }

        Scalar get_v_lower_bound() const override {
            return 0.0;
        }

        Scalar get_u_upper_bound() const override {
            return 1.0;
        }

        Scalar get_v_upper_bound() const override {
            return 1.0;
        }


    public:
        IsoCurveU compute_iso_curve_u(Scalar v) const {
            const int degree_u = Base::get_degree_u();
            const int degree_v = Base::get_degree_v();
            typename IsoCurveU::ControlPoints control_points_u(degree_u+1, _dim);
            for (int i=0; i<=degree_u; i++) {
                typename IsoCurveV::ControlPoints control_points_v(degree_v+1, _dim);
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
            const int degree_u = Base::get_degree_u();
            const int degree_v = Base::get_degree_v();
            typename IsoCurveV::ControlPoints control_points_v(degree_v+1, _dim);
            for (int j=0; j<=degree_v; j++) {
                typename IsoCurveU::ControlPoints control_points_u(degree_u+1, _dim);
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

        BezierPatch<_Scalar, _dim, -1, -1> compute_du_patch() const {
            const int degree_u = Base::get_degree_u();
            const int degree_v = Base::get_degree_v();

            if (degree_u == 0) {
                return BezierPatch<_Scalar, _dim, -1, -1>::ZeroPatch();
            }

            ControlGrid du_grid(degree_u*(degree_v+1), _dim);
            for (int i=0; i<=degree_u-1; i++) {
                for (int j=0; j<=degree_v; j++) {
                    const int row_id = i*(degree_v+1)+j;
                    du_grid.row(row_id) = degree_u *
                        (get_control_point(i+1,j) - get_control_point(i,j));
                }
            }

            BezierPatch<_Scalar, _dim, -1, -1> du_patch;
            du_patch.set_degree_u(degree_u-1);
            du_patch.set_degree_v(degree_v);
            du_patch.swap_control_grid(du_grid);
            du_patch.initialize();
            return du_patch;
        }

        BezierPatch<_Scalar, _dim, -1, -1> compute_dv_patch() const {
            const int degree_u = Base::get_degree_u();
            const int degree_v = Base::get_degree_v();

            if (degree_v == 0) {
                return BezierPatch<_Scalar, _dim, -1, -1>::ZeroPatch();
            }

            ControlGrid dv_grid((degree_u+1)*degree_v, _dim);
            for (int i=0; i<=degree_u; i++) {
                for (int j=0; j<=degree_v-1; j++) {
                    const int row_id = i*degree_v+j;
                    dv_grid.row(row_id) = degree_v *
                        (get_control_point(i,j+1) - get_control_point(i,j));
                }
            }

            BezierPatch<_Scalar, _dim, -1, -1> dv_patch;
            dv_patch.set_degree_u(degree_u);
            dv_patch.set_degree_v(degree_v-1);
            dv_patch.swap_control_grid(dv_grid);
            dv_patch.initialize();
            return dv_patch;
        }

        BezierPatch<_Scalar, _dim, -1, -1> compute_duv_patch() const {
            const int degree_u = Base::get_degree_u();
            const int degree_v = Base::get_degree_v();

            if (degree_u == 0 || degree_v == 0) {
                return BezierPatch<_Scalar, _dim, -1, -1>::ZeroPatch();
            }

            ControlGrid duv_grid(degree_u*degree_v, _dim);
            for (int i=0; i<=degree_u-1; i++) {
                for (int j=0; j<=degree_v-1; j++) {
                    const int row_id = i*degree_v+j;
                    const Point pu0 = degree_u *
                        (get_control_point(i+1,j) - get_control_point(i,j));
                    const Point pu1 = degree_u *
                        (get_control_point(i+1,j+1) - get_control_point(i,j+1));
                    duv_grid.row(row_id) = degree_v * (pu1 - pu0);
                }
            }

            BezierPatch<_Scalar, _dim, -1, -1> duv_patch;
            duv_patch.set_degree_u(degree_u-1);
            duv_patch.set_degree_v(degree_v-1);
            duv_patch.swap_control_grid(duv_grid);
            duv_patch.initialize();
            return duv_patch;
        }

    protected:
        Point get_control_point(int ui, int vj) const {
            const auto degree_v = Base::get_degree_v();
            const auto row_size = degree_v + 1;
            return Base::m_control_grid.row(ui*row_size + vj);
        }
};

}
