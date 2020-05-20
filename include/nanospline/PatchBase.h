#pragma once

#include <array>
#include <cassert>
#include <Eigen/Core>
#include <nanospline/Exceptions.h>
#include <iostream>
using namespace std;
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
        virtual Point get_control_point(int i, int j) const =0;
        virtual int num_control_points_u() const = 0;
        virtual int num_control_points_v() const = 0;


    public:
        virtual UVPoint inverse_evaluate(const Point& p,
                const Scalar min_u,
                const Scalar max_u,
                const Scalar min_v,
                const Scalar max_v) const {
            constexpr Scalar TOL =
                std::numeric_limits<Scalar>::epsilon() * 100;
            const int num_samples = std::max(m_degree_u, m_degree_v) + 1;
            UVPoint uv = approximate_inverse_evaluate(p, num_samples,
                    min_u, max_u, min_v, max_v);
            return newton_raphson(p, uv, 20, TOL,
                    min_u, max_u, min_v, max_v);
        }

    public:
        int num_control_points() const{
            return num_control_points_u() * num_control_points_v();
        }

        bool in_domain_u(Scalar u) const {
            constexpr Scalar eps = std::numeric_limits<Scalar>::epsilon();
            const Scalar u_min = get_u_lower_bound();
            const Scalar u_max = get_u_upper_bound();
            return (u >= u_min - eps) && (u <= u_max + eps);
        }

        bool in_domain_v(Scalar v) const {
            constexpr Scalar eps = std::numeric_limits<Scalar>::epsilon();
            const Scalar v_min = get_v_lower_bound();
            const Scalar v_max = get_v_upper_bound();
            return (v >= v_min - eps) && (v <= v_max + eps);
        }
        bool is_endpoint_u(Scalar u){
            return u == get_u_lower_bound() || u == get_u_upper_bound();
        }
        
        bool is_endpoint_v(Scalar v){
            return v == get_v_lower_bound() || v == get_v_upper_bound();
        }
        bool in_domain(Scalar u, Scalar v) const {
            return in_domain_u(u) && in_domain_v(v);
        }

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

        bool is_split_input_valid(Scalar u, Scalar v) const{
            if(!in_domain(u,v)) {
                throw invalid_setting_error("Parameter not inside of the domain.");
            }
            if(is_endpoint_u(u) || is_endpoint_v(v)){
                return false;
            } else {
                return true;
            }
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
            // 1. find closest control point
            // TODO implement get_control_point(i,j)
            
            // Control points c_{i+/-1,j+/-1} bound the domain
            // 2. find subdomain corresponding to control point subdomain boundary
            // TODO implement get_greville_abcissa
            
            // 3. split curve to find ctrl points on subdomain
            // TODO implement patch split in parent class
            
            // repeat recursively
           
            UVPoint uv(min_u, min_v);
            Scalar min_dist = std::numeric_limits<Scalar>::max();
            for (int i=0; i<=num_samples; i++) {
                const Scalar u = i * (max_u-min_u) / num_samples + min_u;
                for (int j=0; j<=num_samples; j++) {
                    const Scalar v = j * (max_v-min_v) / num_samples + min_v;
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

        UVPoint newton_raphson(
                const Point& p,
                const UVPoint uv,
                const int num_iterations,
                const Scalar tol,
                const Scalar min_u,
                const Scalar max_u,
                const Scalar min_v,
                const Scalar max_v) const {
            Scalar u = uv[0];
            Scalar v = uv[1];
            UVPoint prev_uv = uv;
            Scalar prev_dist = std::numeric_limits<Scalar>::max();
            for (int i=0; i<num_iterations; i++) {
                const Point r = this->evaluate(u, v) - p;
                const Scalar dist = r.norm();
                if (dist < tol) {
                    break;
                }
                 // Converges ok if left alone, could be a result of a poor
                 // initial guess
                if (dist > prev_dist) {
                    // Ops, Newton Raphson diverged...
                    // Use the best result so far.
                    return prev_uv;
                }
                prev_dist = dist;
                prev_uv = {u, v};

                const Point Su = this->evaluate_derivative_u(u, v);
                const Point Sv = this->evaluate_derivative_v(u, v);
                const Point Suu = this->evaluate_2nd_derivative_uu(u, v);
                const Point Svv = this->evaluate_2nd_derivative_vv(u, v);
                const Point Suv = this->evaluate_2nd_derivative_uv(u, v);

                Eigen::Matrix<Scalar, 2, 2> J;
                J << Su.squaredNorm() + r.dot(Suu), Su.dot(Sv) + r.dot(Suv),
                     Sv.dot(Su) + r.dot(Suv), Sv.squaredNorm() + r.dot(Svv);
                Eigen::Matrix<Scalar, 2, 1> kappa;
                kappa << -r.dot(Su), -r.dot(Sv);

                Eigen::Matrix<Scalar, 2, 1> delta = J.inverse() * kappa;

                u += delta[0];
                v += delta[1];

                if (u < min_u) u = min_u;
                if (u > max_u) u = max_u;
                if (v < min_v) v = min_v;
                if (v > max_v) v = max_v;
            }
            return {u, v};
        }

    protected:
        int m_degree_u = -1;
        int m_degree_v = -1;
        ControlGrid m_control_grid;
};

}
