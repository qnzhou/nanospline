#pragma once

#include <Eigen/Core>
#include <iostream>

#include <nanospline/CurveBase.h>

namespace nanospline {

template<typename _Scalar, int _dim, int _degree, bool _generic>
class BSplineBase : public CurveBase<_Scalar, _dim> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        static_assert(_dim > 0, "Dimension must be positive.");
        static_assert(_degree>=0 || _generic,
                "Invalid degree for non-generic B-spline setting");
        using Base = CurveBase<_Scalar, _dim>;
        using Scalar = _Scalar;
        using Point = Eigen::Matrix<Scalar, 1, _dim>;
        using ControlPoints = Eigen::Matrix<Scalar, Eigen::Dynamic, _dim>;
        using KnotVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    public:
        virtual ~BSplineBase()=default;
        virtual Point evaluate(Scalar t) const override =0;
        virtual Scalar inverse_evaluate(const Point& p) const override =0;
        virtual Point evaluate_derivative(Scalar t) const override=0;
        virtual Point evaluate_2nd_derivative(Scalar t) const override=0;

        virtual Scalar approximate_inverse_evaluate(const Point& p,
                const Scalar lower=0.0,
                const Scalar upper=1.0,
                const int level=3) const override {
            assert(in_domain(lower));
            assert(in_domain(upper));
            assert(lower < upper);
            validate_curve();
            const int num_samples = 2 * (get_degree() + 1);

            auto curr_span = locate_span(lower);

            Scalar min_t = 0.0, min_delta = 0.0;
            Scalar min_dist = std::numeric_limits<Scalar>::max();

            Scalar curr_lower = lower;
            while (curr_lower < upper) {
                Scalar curr_upper = std::min(m_knots[curr_span+1], upper);

                if (curr_upper > curr_lower) {
                    const Scalar delta = (curr_upper - curr_lower) / num_samples;
                    for (int j=0; j<num_samples+1; j++) {
                        const Scalar r = (Scalar)(j) / (Scalar)(num_samples);
                        const Scalar t = curr_lower + r * (curr_upper - curr_lower);
                        const auto q = this->evaluate(t);
                        const auto dist = (p-q).squaredNorm();
                        if (dist < min_dist) {
                            min_dist = dist;
                            min_t = t;
                            min_delta = delta;
                        }
                    }
                }

                curr_lower = curr_upper;
                curr_span++;
            }

            if (level <= 0) {
                return min_t;
            } else {
                return approximate_inverse_evaluate(
                        p,
                        std::max(lower, min_t-min_delta),
                        std::min(upper, min_t+min_delta),
                        level-1);
            }
        }

        virtual std::vector<Scalar> compute_inflections(
                const Scalar lower=0.0,
                const Scalar upper=1.0) const override {
            throw not_implemented_error(
                    "Inflection computation is not support for this curve type");
        }

        virtual std::vector<Scalar> reduce_turning_angle(
                const Scalar lower,
                const Scalar upper) const override {
            throw not_implemented_error(
                    "Turning angle reduction is not support for this curve type");
        }

        virtual std::vector<Scalar> compute_singularities(
                const Scalar lower=0.0,
                const Scalar upper=1.0) const override {
            throw not_implemented_error(
                    "Compute singularity is not support for this curve type");
        }

        virtual void insert_knot(Scalar t, int num_copies=1) {
            assert(num_copies >= 1);
            assert(in_domain(t));
            validate_curve();
            const int r = num_copies;
            const int p = get_degree();
            const int k = locate_span(t);
            const int s = (t == m_knots[k]) ? get_multiplicity(k):0;
            assert(k>=p);
            assert(r+s<=p);

            const int n = static_cast<int>(m_control_points.rows()-1);
            const int m = static_cast<int>(m_knots.rows()-1);

            KnotVector knots_new(m+r+1, 1);
            knots_new.segment(0, k+1) = m_knots.segment(0, k+1);
            knots_new.segment(k+1, r).setConstant(t);
            knots_new.segment(k+1+r, m-k-1) = m_knots.segment(k+1,m-k-1);
            knots_new[m+r] = m_knots[m]; // Copy the last knot over.  It has no effects on the curve.

            ControlPoints ctrl_pts_new(n+r+1, _dim);
            assert(k-p+1 >= 0);
            if (k-p+1 > 0) {
                ctrl_pts_new.topRows(k-p+1) =
                    m_control_points.topRows(k-p+1);
            }
            assert(n-k+s+1 >= 0);
            if (n-k+s+1 > 0) {
                ctrl_pts_new.bottomRows(n-k+s+1) =
                    m_control_points.bottomRows(n-k+s+1);
            }

            ControlPoints Rw = m_control_points.block(k-p, 0, p-s+1, _dim);
            for (int j=1; j<=r; j++) {
                int L = k-p+j;
                for (int i=0; i<=p-j-s; i++) {
                    const Scalar diff = m_knots[i+k+1] - m_knots[L+i];
                    Scalar alpha = 0.0;
                    if (diff > 0) {
                        alpha = (t - m_knots[L+i]) / diff;
                    }
                    Rw.row(i) = alpha * Rw.row(i+1) + (1.0-alpha) * Rw.row(i);
                }
                ctrl_pts_new.row(L) = Rw.row(0);
                ctrl_pts_new.row(k+r-j-s) = Rw.row(p-j-s);
            }

            for (int i=k-p+r; i<k-s; i++) {
                ctrl_pts_new.row(i) = Rw.row(i-k+p-r);
            }

            m_control_points.swap(ctrl_pts_new);
            m_knots.swap(knots_new);
        }

    public:
        int locate_span(const Scalar t) const {
            const auto p = get_degree();
            const auto num_knots = m_knots.rows();
            assert(num_knots > m_control_points.rows());
            int low = p;
            int high = static_cast<int>(m_knots.rows()-p-1);
            assert(m_knots[low] <= t);
            assert(m_knots[high] >= t);

            auto bypass_duplicates_after = [this, num_knots](int i) {
                while(i+1<num_knots && this->m_knots[i] == this->m_knots[i+1]) {
                    i=i+1;
                }
                return i;
            };

            auto bypass_duplicates_before= [this](int i) {
                while(i-1>=0 && this->m_knots[i] == this->m_knots[i-1]) {
                    i=i-1;
                }
                return i;
            };

            if (t == m_knots[high]) return bypass_duplicates_before(high)-1;

            int mid = (high+low) / 2;
            while(t < m_knots[mid] || t >= m_knots[mid+1]) {
                if (t < m_knots[mid]) high=mid;
                else low = mid;
                mid = (high+low) / 2;
            }

            return bypass_duplicates_after(mid);
        }

        int get_multiplicity(int k) const {
            const int m = static_cast<int>(m_knots.rows());
            int s =1;
            for (int i=k-1; i>=0; i--) {
                if (m_knots[i] == m_knots[k]) s++;
                else break;
            }
            for (int i=k+1; i<m; i++) {
                if (m_knots[i] == m_knots[k]) s++;
                else break;
            }
            return s;
        }

        const ControlPoints& get_control_points() const {
            return m_control_points;
        }

        template<typename Derived>
        void set_control_points(const Eigen::PlainObjectBase<Derived>& ctrl_pts) {
            m_control_points = ctrl_pts;
        }

        template<typename Derived>
        void set_control_points(Eigen::PlainObjectBase<Derived>&& ctrl_pts) {
            m_control_points.swap(ctrl_pts);
        }

        const KnotVector& get_knots() const {
            return m_knots;
        }

        template<typename Derived>
        void set_knots(const Eigen::PlainObjectBase<Derived>& knots) {
            m_knots = knots;
        }

        template<typename Derived>
        void set_knots(Eigen::PlainObjectBase<Derived>&& knots) {
            m_knots.swap(knots);
        }

        int get_degree() const {
            return static_cast<int>(
                    m_knots.rows() - m_control_points.rows() - 1);
        }

        bool in_domain(Scalar t) const {
            const int p = get_degree();
            const int num_knots = static_cast<int>(m_knots.rows());
            const Scalar t_min = m_knots[p];
            const Scalar t_max = m_knots[num_knots-p-1];
            return (t >= t_min) && (t <= t_max);
        }

        Scalar get_domain_lower_bound() const {
            const int p = get_degree();
            return m_knots[p];
        }

        Scalar get_domain_upper_bound() const {
            const int p = get_degree();
            const int num_knots = static_cast<int>(m_knots.rows());
            return m_knots[num_knots-p-1];
        }

        virtual void write(std::ostream &out) const override {
            out << "c:\n" << m_control_points << "\n";
            out << "k:\n" << m_knots << "\n";
        }

    protected:
        void validate_curve() const {
            const auto d = get_degree();
            if (d < 0 || (_degree >= 0 && d != _degree)) {
                throw invalid_setting_error("Invalid BSpline curve: wrong degree.");
            }
        }

    protected:
        ControlPoints m_control_points;
        KnotVector m_knots;
};

}
