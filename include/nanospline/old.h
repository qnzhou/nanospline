#pragma once

#include <iostream>
#include <Eigen/Core>

namespace nanospline {

template<typename Scalar, size_t dim=3, int order=3, int N=order, bool rational=false,
    bool _dynamic = order<0,
    bool _is_bezier = order==N>
struct BSplineTrait { };

template<typename Scalar, size_t dim, int order>
struct BSplineTrait<Scalar, dim, order, order, false, false, true> {
    // Non-rational, static, bezier
    using Point = Eigen::Matrix<Scalar, 1, dim>;
    using ControlPoints = Eigen::Matrix<Scalar, order+1, dim>;
    using KnotVector = void;
    using WeightVector = void;
};

template<typename Scalar, size_t dim, int order>
struct BSplineTrait<Scalar, dim, order, order, true, false, true> {
    // Rational, static, bezier
    using Point = Eigen::Matrix<Scalar, 1, dim+1>;
    using ControlPoints = Eigen::Matrix<Scalar, order+1, dim+1>;
    using KnotVector = void;
    using WeightVector = Eigen::Matrix<Scalar, 1, order+1>;
};

template<typename Scalar, size_t dim, int order>
struct BSplineTrait<Scalar, dim, order, order, false, true, true> {
    // Non-rational, dynamic, bezier
    using Point = Eigen::Matrix<Scalar, 1, dim>;
    using ControlPoints = Eigen::Matrix<Scalar, Eigen::Dynamic, dim>;
    using KnotVector = void;
    using WeightVector = void;
};

template<typename Scalar, size_t dim, int order>
struct BSplineTrait<Scalar, dim, order, order, true, true, true> {
    // Rational, dynamic, bezier
    using Point = Eigen::Matrix<Scalar, 1, dim+1>;
    using ControlPoints = Eigen::Matrix<Scalar, Eigen::Dynamic, dim+1>;
    using KnotVector = void;
    using WeightVector = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
};

template<typename Scalar, size_t dim, int order, int N>
struct BSplineTrait<Scalar, dim, order, N, false, false, false> {
    // Non-rational, static, bspline
    using Point = Eigen::Matrix<Scalar, 1, dim>;
    using ControlPoints = Eigen::Matrix<Scalar, N+1, dim>;
    using KnotVector = Eigen::Matrix<Scalar, 1, N+order+2>;
    using WeightVector = void;
};

template<typename Scalar, size_t dim, int order, int N>
struct BSplineTrait<Scalar, dim, order, N, true, false, false> {
    // Rational, static, bspline
    using Point = Eigen::Matrix<Scalar, 1, dim+1>;
    using ControlPoints = Eigen::Matrix<Scalar, N+1, dim+1>;
    using KnotVector = Eigen::Matrix<Scalar, 1, N+order+2>;
    using WeightVector = Eigen::Matrix<Scalar, 1, N+1>;
};

template<typename Scalar, size_t dim, int order, int N>
struct BSplineTrait<Scalar, dim, order, N, false, true, false> {
    // Non-rational, dynamic, bspline
    using Point = Eigen::Matrix<Scalar, 1, dim>;
    using ControlPoints = Eigen::Matrix<Scalar, Eigen::Dynamic, dim>;
    using KnotVector = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
    using WeightVector = void;
};

template<typename Scalar, size_t dim, int order, int N>
struct BSplineTrait<Scalar, dim, order, N, true, true, false> {
    // Rational, dynamic, bezier
    using Point = Eigen::Matrix<Scalar, 1, dim+1>;
    using ControlPoints = Eigen::Matrix<Scalar, Eigen::Dynamic, dim+1>;
    using KnotVector = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
    using WeightVector = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
};




template<typename Scalar, size_t dim, int order=3, int N=order, bool rational=false, bool specialize=true>
class BSpline {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    public:
        using Trait = BSplineTrait<Scalar, dim, order, N, rational>;
        using Point = typename Trait::Point;
        using ControlPoints = typename Trait::ControlPoints;
        using KnotVector = typename Trait::KnotVector;
        using WeightVector = typename Trait::WeightVector;

    public:
        BSpline() {
            static_assert(order <= N, "Invalid order and N combination");
        }

    public:
        Point evaluate(Scalar t) const {
        }

        Scalar project(const Point& p) const {
        }

    public:
        template<typename Derived>
        void set_control_points(const Eigen::PlainObjectBase<Derived>& pts) {
            m_control_points = pts;
        }

        template<typename Derived>
        void set_control_points(const Eigen::PlainObjectBase<Derived>&& pts) {
            m_control_points.swap(pts);
        }

        template<typename Derived>
        void knot_vector(const Eigen::PlainObjectBase<Derived>& knots) {
            static_assert(order < N, "Cannot set knots because Curve is Bezier");
            m_knots = knots;
        }

        template<typename Derived>
        void knot_vector(const Eigen::PlainObjectBase<Derived>&& knots) {
            static_assert(order < N, "Cannot set knots because Curve is Bezier");
            m_knots.swap(knots);
        }

        template<typename Derived>
        void set_weights(const Eigen::PlainObjectBase<Derived>& weights) {
            static_assert(rational, "Only rational spline requires weights.");
            m_weights = weights;
        }

        template<typename Derived>
        void set_weights(const Eigen::PlainObjectBase<Derived>&& weights) {
            static_assert(rational, "Only rational spline requires weights.");
            m_weights.swap(weights);
        }


    public:
        int test() const {
            return test_impl(std::integral_constant<bool, order == N>());
        }

    private:
        int test_impl(std::true_type) const {
            return 0;
        }
        int test_impl(std::false_type) const {
            return 1;
        }

    private:
        ControlPoints m_control_points;
        KnotVector m_knots;
        WeightVector m_weights;
};

}
