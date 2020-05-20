#pragma once

#include <nanospline/PatchBase.h>
#include <nanospline/NURBS.h>
#include <nanospline/BSplinePatch.h>

namespace nanospline {

template<typename _Scalar, int _dim=3,
    int _degree_u=3, int _degree_v=3>
class NURBSPatch final : public PatchBase<_Scalar, _dim> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        static_assert(_dim > 0, "Dimension must be positive.");

        using Base = PatchBase<_Scalar, _dim>;
        using Scalar = typename Base::Scalar;
        using Point = typename Base::Point;
        using ControlGrid = typename Base::ControlGrid;
        using Weights = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        using KnotVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        using BSplinePatchHomogeneous = BSplinePatch<Scalar, _dim+1, _degree_u, _degree_v>;
        using IsoCurveU = NURBS<Scalar, _dim, _degree_u>;
        using IsoCurveV = NURBS<Scalar, _dim, _degree_v>;

    public:
        NURBSPatch() {
            Base::set_degree_u(_degree_u);
            Base::set_degree_v(_degree_v);
        }

    public:
        int num_control_points_u() const override {
            return m_homogeneous.num_control_points_u();
        }
        int num_control_points_v() const override {
            return m_homogeneous.num_control_points_v();
        }
        Point get_control_point(int i, int j) const override {
            Eigen::Matrix<Scalar, 1, _dim+1> control_point = m_homogeneous.get_control_point(i,j);
            return control_point.head(_dim);
        }

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

        Point evaluate_2nd_derivative_uu(Scalar u, Scalar v) const override {
            validate_initialization();
            constexpr Scalar tol = std::numeric_limits<Scalar>::epsilon();
            const auto p0 = m_homogeneous.evaluate(u, v);
            const auto du = m_homogeneous.evaluate_derivative_u(u, v);
            const auto duu = m_homogeneous.evaluate_2nd_derivative_uu(u, v);

            const Point Auu = duu.template head<_dim>();
            const Scalar wuu = duu[_dim];
            const Scalar w = p0[_dim];
            if (w <= tol) return Point::Zero();

            const Point S = p0.template head<_dim>() / w;
            const Point Su = evaluate_derivative_u(u, v);
            const Scalar wu = du[_dim];

            return (Auu - Su * wu * 2 - S * wuu) / w;
        }

        Point evaluate_2nd_derivative_vv(Scalar u, Scalar v) const override {
            validate_initialization();
            constexpr Scalar tol = std::numeric_limits<Scalar>::epsilon();

            const auto p0 = m_homogeneous.evaluate(u, v);
            const auto dv = m_homogeneous.evaluate_derivative_v(u, v);
            const auto dvv = m_homogeneous.evaluate_2nd_derivative_vv(u, v);

            const Point Avv = dvv.template segment<_dim>(0);
            const Scalar wvv = dvv[_dim];
            const Scalar w = p0[_dim];
            if (w <= tol) return Point::Zero();

            const Point S = p0.template segment<_dim>(0) / w;
            const Point Sv = evaluate_derivative_v(u, v);
            const Scalar wv = dv[_dim];

            return (Avv - Sv * wv * 2 - S * wvv) / w;
        }

        Point evaluate_2nd_derivative_uv(Scalar u, Scalar v) const override {
            validate_initialization();
            constexpr Scalar tol = std::numeric_limits<Scalar>::epsilon();

            const auto p0 = m_homogeneous.evaluate(u, v);
            const auto du = m_homogeneous.evaluate_derivative_u(u, v);
            const auto dv = m_homogeneous.evaluate_derivative_v(u, v);
            const auto duv = m_homogeneous.evaluate_2nd_derivative_uv(u, v);

            const Point Auv = duv.template segment<_dim>(0);
            const Scalar wuv = duv[_dim];
            const Point S = p0.template segment<_dim>(0) / p0[_dim];
            const Scalar w = p0[_dim];
            if (w <= tol) return Point::Zero();

            const Scalar wu = du[_dim];
            const Scalar wv = dv[_dim];
            const Point Su = evaluate_derivative_u(u, v);
            const Point Sv = evaluate_derivative_v(u, v);

            return (Auv - S*wuv - wu*Sv - wv*Su) / w;
        }

        void initialize() override {
            const auto num_control_pts = Base::m_control_grid.rows();
            if (m_weights.size() != num_control_pts) {
                throw invalid_setting_error("Weights and control grid mismatch");
            }

            typename BSplinePatchHomogeneous::ControlGrid ctrl_pts(
                    num_control_pts, _dim+1);
            ctrl_pts.template leftCols<_dim>() =
                Base::m_control_grid.array().colwise() * m_weights.array();
            ctrl_pts.template rightCols<1>() = m_weights;

            m_homogeneous.set_control_grid(std::move(ctrl_pts));
            m_homogeneous.set_knots_u(m_knots_u);
            m_homogeneous.set_knots_v(m_knots_v);
            m_homogeneous.set_degree_u(Base::get_degree_u());
            m_homogeneous.set_degree_v(Base::get_degree_v());
            m_homogeneous.initialize();
        }

        Scalar get_u_lower_bound() const override {
            const int degree_u = Base::get_degree_u();
            return m_knots_u[degree_u];
        }

        Scalar get_v_lower_bound() const override {
            const int degree_v = Base::get_degree_v();
            return m_knots_v[degree_v];
        }

        Scalar get_u_upper_bound() const override {
            const auto num_knots = static_cast<int>(m_knots_u.rows());
            const int degree_u = Base::get_degree_u();
            return m_knots_u[num_knots-degree_u-1];
        }

        Scalar get_v_upper_bound() const override {
            const auto num_knots = static_cast<int>(m_knots_v.rows());
            const int degree_v = Base::get_degree_v();
            return m_knots_v[num_knots-degree_v-1];
        }

    public:
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

        const BSplinePatchHomogeneous& get_homogeneous() const {
            return m_homogeneous;
        }

    private:
        void validate_initialization() const {
            const auto& ctrl_pts = m_homogeneous.get_control_grid();
            if (ctrl_pts.rows() != Base::m_control_grid.rows() ||
                ctrl_pts.rows() != m_weights.rows() ) {
                throw invalid_setting_error("NURBS patch is not initialized.");
            }
        }

    protected:
        Weights m_weights;
        BSplinePatchHomogeneous m_homogeneous;
        KnotVector m_knots_u;
        KnotVector m_knots_v;

};

}
