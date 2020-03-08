#pragma once

#include <nanospline/PatchBase.h>
#include <nanospline/BSpline.h>

namespace nanospline {

template<typename _Scalar, int _dim=3,
    int _degree_u=3, int _degree_v=3>
class BSplinePatch : PatchBase<_Scalar, _dim> {
    public:
        static_assert(_dim > 0, "Dimension must be positive.");
        static_assert(_degree_u >= 0, "Degree in u must be positive.");
        static_assert(_degree_v >= 0, "Degree in v must be positive.");

        using Base = PatchBase<_Scalar, _dim>;
        using Scalar = typename Base::Scalar;
        using Point = typename Base::Point;
        using ControlGrid = Eigen::Matrix<Scalar, (_degree_u+1)*(_degree_v+1), _dim>;
        using KnotVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        using IsoCurveU = BSpline<Scalar, _dim, _degree_u>;
        using IsoCurveV = BSpline<Scalar, _dim, _degree_v>;

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

            typename IsoCurveU::ControlPoints control_points_u(
                    num_u_knots-_degree_u-1, _dim);
            for (int i=0; i<num_u_knots-_degree_u-1; i++) {
                typename IsoCurveV::ControlPoints control_points_v(
                        num_v_knots-_degree_v-1, _dim);
                for (int j=0; j<num_v_knots-_degree_v-1; j++) {
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

            typename IsoCurveV::ControlPoints control_points_v(
                    num_v_knots-_degree_v-1, _dim);
            for (int j=0; j<num_v_knots-_degree_v-1; j++) {
                typename IsoCurveU::ControlPoints control_points_u(
                        num_u_knots-_degree_u-1, _dim);
                for (int i=0; i<num_u_knots-_degree_u-1; i++) {
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
            return m_knots_u[_degree_u];
        }

        Scalar get_v_lower_bound() const {
            return m_knots_v[_degree_v];
        }

        Scalar get_u_upper_bound() const {
            const auto num_knots = static_cast<int>(m_knots_u.rows());
            return m_knots_u[num_knots-_degree_u-1];
        }

        Scalar get_v_upper_bound() const {
            const auto num_knots = static_cast<int>(m_knots_v.rows());
            return m_knots_v[num_knots-_degree_v-1];
        }

    protected:
        Point get_control_point(int ui, int vj) const {
            const auto row_size = m_knots_u.size() - _degree_u - 1;
            return m_control_grid.row(vj*row_size + ui);
        }

    protected:
        ControlGrid m_control_grid;
        KnotVector m_knots_u;
        KnotVector m_knots_v;
};

}
