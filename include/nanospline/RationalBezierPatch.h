#pragma once

#include <Eigen/Core>
#include <nanospline/PatchBase.h>
#include <nanospline/RationalBezier.h>
#include <nanospline/BezierPatch.h>

namespace nanospline {

template<typename _Scalar, int _dim=3,
    int _degree_u=3, int _degree_v=3>
class RationalBezierPatch final : public PatchBase<_Scalar, _dim> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        static_assert(_dim > 0, "Dimension must be positive.");

        using Base = PatchBase<_Scalar, _dim>;
        using Scalar = typename Base::Scalar;
        using UVPoint = typename Base::UVPoint;
        using Point = typename Base::Point;
        using ThisType = RationalBezierPatch<_Scalar, _dim, _degree_u, _degree_v>;
        using ControlGrid = typename Base::ControlGrid;
        using Weights = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        using BezierPatchHomogeneous = BezierPatch<Scalar, _dim+1, _degree_u, _degree_v>;
        using IsoCurveU = RationalBezier<Scalar, _dim, _degree_u>;
        using IsoCurveV = RationalBezier<Scalar, _dim, _degree_v>;

    public:
        RationalBezierPatch() {
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
        UVPoint get_control_point_preimage(int i, int j) const override {
            return m_homogeneous.get_control_point_preimage(i, j);
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
            typename BezierPatchHomogeneous::ControlGrid ctrl_pts(
                    Base::m_control_grid.rows(), _dim+1);
            ctrl_pts.template leftCols<_dim>() =
                Base::m_control_grid.array().colwise() * m_weights.array();
            ctrl_pts.template rightCols<1>() = m_weights;

            m_homogeneous.set_control_grid(std::move(ctrl_pts));
            m_homogeneous.set_degree_u(Base::get_degree_u());
            m_homogeneous.set_degree_v(Base::get_degree_v());
            m_homogeneous.initialize();
        }
        
        void set_homogeneous(const BezierPatchHomogeneous& homogeneous) {
            const auto ctrl_pts = homogeneous.get_control_grid();
            m_homogeneous = homogeneous;
            m_weights = ctrl_pts.template rightCols<1>();
            Base::m_control_grid =
                ctrl_pts.template leftCols<_dim>().array().colwise()
                / m_weights.array();
            validate_initialization();
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
        
        std::vector<ThisType> split_u(Scalar u){
            const auto parts = m_homogeneous.split_u(u);
            std::vector<ThisType> results;
            results.reserve(2);
            for (const auto& c : parts) {
                results.emplace_back();
                results.back().set_homogeneous(c);
            }
            return results;

        }
        
        std::vector<ThisType> split_v(Scalar v){
            const auto parts = m_homogeneous.split_v(v);
            std::vector<ThisType> results;
            results.reserve(2);
            for (const auto& c : parts) {
                results.emplace_back();
                results.back().set_homogeneous(c);
            }
            return results;

        }
        
        std::vector<ThisType> split(Scalar u, Scalar v){
            const auto parts = m_homogeneous.split(u,v);
            std::vector<ThisType> results;
            results.reserve(4);
            for (const auto& c : parts) {
                results.emplace_back();
                results.back().set_homogeneous(c);
            }
            return results;

        }
        
        UVPoint approximate_inverse_evaluate(const Point& p,
                const int num_samples,
                const Scalar min_u,
                const Scalar max_u,
                const Scalar min_v,
                const Scalar max_v,
                const int level=3) const {
            
            // 1. find closest control point
            Scalar min_dist = std::numeric_limits<Scalar>::max();
            int i_min, j_min;
            for (int ui = 0; ui < num_control_points_u(); ui++) {
              for (int vj = 0; vj < num_control_points_v(); vj++) {
                  Point control_point = get_control_point(ui, vj);
                 const auto dist = (p - control_point).squaredNorm();
                 cout << dist << ", " << min_dist << endl;
                 //cout << "dist < min_dist: " << int(dist < min_dist )<< endl;

                 if(dist < min_dist){
                     min_dist = dist;
                     i_min = ui;
                     j_min = vj;

                 }
              }
            }
            if (level <= 0) {
                return get_control_point_preimage(i_min, j_min);

            } else {

            // Control points c_{i+/-1,j+/-1} bound the domain
            // 2. find subdomain corresponding to control point subdomain boundary
            UVPoint uv_min = get_control_point_preimage(
                    i_min > 0 ? i_min-1: i_min,
                    j_min > 0 ? j_min-1: j_min);
            UVPoint uv_max = get_control_point_preimage(
                    i_min < num_control_points_u() ? i_min+1 : i_min,
                    j_min < num_control_points_v() ? j_min+1 : j_min);
            cout << "min, max:  " << uv_min << ", " << uv_max << endl;
            // 3. split curve to find ctrl points on subdomain
            // TODO implement patch split in parent class
            
            // repeat recursively
            // TODO remap solution up through affine subdomain transformations
                return approximate_inverse_evaluate(p, num_samples,
                        std::max(uv_min(0), min_u),
                        std::min(uv_max(0), max_u),
                        std::max(uv_min(1), min_v),
                        std::min(uv_max(1), max_v),
                        level-1);
            }


        }


    protected:
        std::tuple<Point, Scalar> get_control_point_and_weight(int ui, int vj) const {
            const int degree_u = Base::get_degree_u();
            int index = vj*(degree_u+1) + ui;
            return std::make_tuple(Base::m_control_grid.row(index), m_weights[index]);
        }

        void validate_initialization() const {
            const auto& ctrl_pts = m_homogeneous.get_control_grid();
            if (ctrl_pts.rows() != Base::m_control_grid.rows() ||
                ctrl_pts.rows() != m_weights.rows() ) {
                throw invalid_setting_error("Rational Bezier patch is not initialized.");
            }
        }

    protected:
        Weights m_weights;
        BezierPatchHomogeneous m_homogeneous;
};

}
