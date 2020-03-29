#pragma once

#include <nanospline/PatchBase.h>
#include <nanospline/BSpline.h>

namespace nanospline {

template<typename _Scalar, int _dim=3,
    int _degree_u=3, int _degree_v=3>
class BSplinePatch final : public PatchBase<_Scalar, _dim> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        static_assert(_dim > 0, "Dimension must be positive.");

        using Base = PatchBase<_Scalar, _dim>;
        using Scalar = typename Base::Scalar;
        using Point = typename Base::Point;
        using ControlGrid = typename Base::ControlGrid;
        using KnotVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        using IsoCurveU = BSpline<Scalar, _dim, _degree_u>;
        using IsoCurveV = BSpline<Scalar, _dim, _degree_v>;

    public:
        BSplinePatch() {
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
            const auto num_v_knots = m_knots_v.size();
            const auto num_u_knots = m_knots_u.size();
            const int degree_u = Base::get_degree_u();
            const int degree_v = Base::get_degree_v();
            if (Base::m_control_grid.rows() !=
                    (num_u_knots-degree_u-1) * (num_v_knots-degree_v-1)) {
                throw invalid_setting_error(
                        "Control grid size mismatch uv degrees");
            }
        }

    public:
        const KnotVector& get_knots_u() const {
            return m_knots_u;
        }

        const KnotVector& get_knots_v() const {
            return m_knots_v;
        }

        template<typename Derived>
        void set_knots_u(const Eigen::PlainObjectBase<Derived>& knots) {
            m_knots_u = knots;
        }

        template<typename Derived>
        void set_knots_v(const Eigen::PlainObjectBase<Derived>& knots) {
            m_knots_v = knots;
        }

        template<typename Derived>
        void set_knots_u(Eigen::PlainObjectBase<Derived>&& knots) {
            m_knots_u.swap(knots);
        }

        template<typename Derived>
        void set_knots_v(Eigen::PlainObjectBase<Derived>&& knots) {
            m_knots_v.swap(knots);
        }

        IsoCurveU compute_iso_curve_u(Scalar v) const {
            const auto num_v_knots = m_knots_v.size();
            const auto num_u_knots = m_knots_u.size();
            const int degree_u = Base::get_degree_u();
            const int degree_v = Base::get_degree_v();

            typename IsoCurveU::ControlPoints control_points_u(
                    num_u_knots-degree_u-1, _dim);
            for (int i=0; i<num_u_knots-degree_u-1; i++) {
                typename IsoCurveV::ControlPoints control_points_v(
                        num_v_knots-degree_v-1, _dim);
                for (int j=0; j<num_v_knots-degree_v-1; j++) {
                    control_points_v.row(j) = get_control_point(i, j);
                }

                IsoCurveV iso_curve_v;
                iso_curve_v.set_control_points(std::move(control_points_v));
                iso_curve_v.set_knots(m_knots_v);
                control_points_u.row(i) = iso_curve_v.evaluate(v);
            }

            IsoCurveU iso_curve_u;
            iso_curve_u.set_control_points(std::move(control_points_u));
            iso_curve_u.set_knots(m_knots_u);
            return iso_curve_u;
        }

        IsoCurveV compute_iso_curve_v(Scalar u) const {
            const auto num_v_knots = m_knots_v.size();
            const auto num_u_knots = m_knots_u.size();
            const int degree_u = Base::get_degree_u();
            const int degree_v = Base::get_degree_v();

            typename IsoCurveV::ControlPoints control_points_v(
                    num_v_knots-degree_v-1, _dim);
            for (int j=0; j<num_v_knots-degree_v-1; j++) {
                typename IsoCurveU::ControlPoints control_points_u(
                        num_u_knots-degree_u-1, _dim);
                for (int i=0; i<num_u_knots-degree_u-1; i++) {
                    control_points_u.row(i) = get_control_point(i, j);
                }

                IsoCurveU iso_curve_u;
                iso_curve_u.set_control_points(std::move(control_points_u));
                iso_curve_u.set_knots(m_knots_u);
                control_points_v.row(j) = iso_curve_u.evaluate(u);
            }

            IsoCurveV iso_curve_v;
            iso_curve_v.set_control_points(std::move(control_points_v));
            iso_curve_v.set_knots(m_knots_v);
            return iso_curve_v;
        }

        Scalar get_u_lower_bound() const {
            const auto degree_u = Base::get_degree_u();
            return m_knots_u[degree_u];
        }

        Scalar get_v_lower_bound() const {
            const auto degree_v = Base::get_degree_v();
            return m_knots_v[degree_v];
        }

        Scalar get_u_upper_bound() const {
            const auto num_knots = static_cast<int>(m_knots_u.rows());
            const auto degree_u = Base::get_degree_u();
            return m_knots_u[num_knots-degree_u-1];
        }

        Scalar get_v_upper_bound() const {
            const auto num_knots = static_cast<int>(m_knots_v.rows());
            const auto degree_v = Base::get_degree_v();
            return m_knots_v[num_knots-degree_v-1];
        }

    protected:
        Point get_control_point(int ui, int vj) const {
            const auto degree_v = Base::get_degree_v();
            const auto row_size = m_knots_v.size() - degree_v - 1;
            return Base::m_control_grid.row(ui*row_size + vj);
        }

    protected:
        KnotVector m_knots_u;
        KnotVector m_knots_v;
};

}
