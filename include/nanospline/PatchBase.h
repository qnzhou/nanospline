#pragma once

#include <array>
#include <cassert>
#include <Eigen/Core>

namespace nanospline {

template<typename _Scalar, int _dim>
class PatchBase {
    public:
        using Scalar = _Scalar;
        using Point = Eigen::Matrix<Scalar, 1, _dim>;
        using UVPoint = Eigen::Matrix<Scalar, 1, 2>;
        using ControlGrid = Eigen::Matrix<Scalar, Eigen::Dynamic, _dim>;

    public:
        virtual ~PatchBase() = default;
        virtual Point evaluate(Scalar u, Scalar v) const =0;
        virtual Point evaluate_derivative_u(Scalar u, Scalar v) const=0;
        virtual Point evaluate_derivative_v(Scalar u, Scalar v) const=0;
        virtual Point evaluate_2nd_derivative_uu(Scalar u, Scalar v) const =0;
        virtual Point evaluate_2nd_derivative_vv(Scalar u, Scalar v) const =0;
        virtual Point evaluate_2nd_derivative_uv(Scalar u, Scalar v) const =0;
        virtual void initialize() =0;
        virtual Scalar get_u_lower_bound() const =0;
        virtual Scalar get_u_upper_bound() const =0;
        virtual Scalar get_v_lower_bound() const =0;
        virtual Scalar get_v_upper_bound() const =0;

    public:
        virtual UVPoint inverse_evaluate(const Point& p,
                const Scalar min_u,
                const Scalar max_u,
                const Scalar min_v,
                const Scalar max_v) const {
            const int num_samples = std::max(m_degree_u, m_degree_v) + 1;
            UVPoint uv = approximate_inverse_evaluate(p, num_samples,
                    min_u, max_u, min_v, max_v);
            return uv;
        }

    public:
        void set_degree_u(int degree) {
            m_degree_u = degree;
        }

        void set_degree_v(int degree) {
            m_degree_v = degree;
        }

        int get_degree_u() const {
            assert(m_degree_u >= 0);
            return m_degree_u;
        }

        int get_degree_v() const {
            assert(m_degree_v >= 0);
            return m_degree_v;
        }

        const ControlGrid& get_control_grid() const {
            return m_control_grid;
        }

        /**
         * Control grid presents a 2D grid of control points.  It is linearized
         * in V-major.  I.e. Let C_uv denotes the control point at (u, v), then
         * control grid = [C_00, C_01, ... C_0n,
         *                 C_10, C_11, ... C_1n,
         *                 ...
         *                 C_m0, C_m1, ... C_mn].
         */
        template<typename Derived>
        void set_control_grid(const Eigen::PlainObjectBase<Derived>& ctrl_grid) {
            m_control_grid = ctrl_grid;
        }

        template<typename Derived>
        void set_control_grid(Eigen::PlainObjectBase<Derived>&& ctrl_grid) {
            m_control_grid.swap(ctrl_grid);
        }

        template<typename Derived>
        void swap_control_grid(Eigen::PlainObjectBase<Derived>& ctrl_grid) {
            m_control_grid.swap(ctrl_grid);
        }

        constexpr int get_dim() const {
            return _dim;
        }

    protected:
        UVPoint approximate_inverse_evaluate(const Point& p,
                const int num_samples,
                const Scalar min_u,
                const Scalar max_u,
                const Scalar min_v,
                const Scalar max_v,
                const int level=3) const {
            UVPoint uv(min_u, min_v);
            Scalar min_dist = std::numeric_limits<Scalar>::max();
            for (int i=0; i<=num_samples; i++) {
                const Scalar u = i * (max_u-min_u) / num_samples + min_u;
                for (int j=0; j<=num_samples; j++) {
                    const Scalar v = i * (max_v-min_v) / num_samples + min_v;
                    const Point q = this->evaluate(u, v);
                    const auto dist = (p-q).squaredNorm();
                    if (dist < min_dist) {
                        min_dist = dist;
                        uv = {u, v};
                    }
                }
            }

            if (level <= 0) {
                return uv;
            } else {
                const auto delta_u = (max_u-min_u) / num_samples;
                const auto delta_v = (max_v-min_v) / num_samples;
                return approximate_inverse_evaluate(p, num_samples,
                        std::max(uv[0]-delta_u, min_u),
                        std::min(uv[0]+delta_u, max_u),
                        std::max(uv[1]-delta_v, min_v),
                        std::min(uv[1]+delta_v, max_v),
                        level-1);
            }
        }

        //UVPoint newton_raphson(
        //        const Point& p,
        //        UVPoint uv,
        //        const int num_iterations,
        //        const Scalar tol,
        //        const Scalar min_u,
        //        const Scalar max_u,
        //        const Scalar min_v,
        //        const Scalar max_v) const {
        //    for (int i=0; i<num_iterations; i++) {
        //        const Point r = this->evaluate(uv[0], uv[1]) - p;
        //    }
        //}

    protected:
        int m_degree_u = -1;
        int m_degree_v = -1;
        ControlGrid m_control_grid;
};

}
