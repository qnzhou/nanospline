#pragma once

#include <nanospline/Exceptions.h>
#include <nanospline/SimplexBase.h>

#include <array>
#include <cassert>
#include <exception>
#include <span>

namespace nanospline {

namespace internal {

/**
 * Compile-time computation of n choose r.
 */
constexpr size_t choose(uint8_t n, uint8_t r)
{
    size_t accum = 1;
    if (n - r > r) r = n - r;
    for (uint8_t x = 1; x <= n - r; ++x) {
        accum *= (r + x);
        accum /= x;
    }
    return accum;
}

template <int N>
constexpr std::array<int, N + 1> factorial()
{
    std::array<int, N + 1> result = {};
    result[0] = 1;
    for (int i = 1; i <= N; i++) {
        result[static_cast<size_t>(i)] = result[static_cast<size_t>(i - 1)] * i;
    }
    return result;
}

template <typename Derived>
void generate_multi_indices(
    uint8_t simplex_dim, uint8_t degree, Eigen::MatrixBase<Derived>& ctrl_net)
{
    using Scalar = typename Derived::Scalar;
    assert(static_cast<size_t>(ctrl_net.rows()) == choose(simplex_dim + degree, degree));

    // Base case.
    if (simplex_dim == 0) {
        ctrl_net.setConstant(degree);
        return;
    }

    size_t count = 0;
    for (uint8_t d = 0; d <= degree; d++) {
        size_t m = choose(simplex_dim + d - 1, d);
        ctrl_net.block(static_cast<Eigen::Index>(count), 0, static_cast<Eigen::Index>(m), 1)
            .setConstant(degree - d);
        // Note: we cannot use Eigen::Block type for layer in recursive call.
        Eigen::Ref<Eigen::Matrix<Scalar, -1, -1, Eigen::RowMajor>> layer =
            ctrl_net.block(static_cast<Eigen::Index>(count),
                1,
                static_cast<Eigen::Index>(m),
                static_cast<Eigen::Index>(simplex_dim));
        generate_multi_indices(simplex_dim - 1, d, layer);
        count += static_cast<size_t>(layer.rows());
    }
}

} // namespace internal

template <typename _Scalar, uint8_t _simplex_dim, uint8_t _degree = 3, int8_t _ordinate_dim = -1>
class BezierSimplex : public SimplexBase<_Scalar, _simplex_dim, _ordinate_dim>
{
public:
    static_assert(_degree >= 0, "Degree must be non-negative");
    using Base = SimplexBase<_Scalar, _simplex_dim, _ordinate_dim>;
    using ThisType = BezierSimplex<_Scalar, _simplex_dim, _degree, _ordinate_dim>;
    using Scalar = typename Base::Scalar;
    using BarycentricPoint = typename Base::BarycentricPoint;
    using Point = typename Base::Point;

protected:
    static constexpr size_t m_num_control_points =
        internal::choose(_degree + _simplex_dim, _simplex_dim);

public:
    using ControlPoints =
        Eigen::Matrix<Scalar, m_num_control_points, _simplex_dim + 1, Eigen::RowMajor>;
    using MultiIndices =
        Eigen::Matrix<uint8_t, m_num_control_points, _simplex_dim + 1, Eigen::RowMajor>;
    using Ordinates = Eigen::Matrix<Scalar, m_num_control_points, _ordinate_dim, Eigen::RowMajor>;

public:
    BezierSimplex() = default;

    constexpr uint8_t get_degree() const { return _degree; }
    constexpr size_t get_num_control_points() const { return m_num_control_points; }

    ControlPoints get_control_points() const
    {
        if constexpr (_degree == 0) {
            return ControlPoints::Zero();
        } else {
            ControlPoints ctrl_pts;
            internal::generate_multi_indices(_simplex_dim, _degree, ctrl_pts);
            return ctrl_pts / static_cast<Scalar>(_degree);
        }
    }

    const Ordinates& get_ordinates() const { return m_ordinates; }
    Ordinates& get_ordinates() { return m_ordinates; }
    void set_ordinates(const Ordinates& ordinates) { m_ordinates = ordinates; }


public:
    Point evaluate(BarycentricPoint b) const
    {
        constexpr auto factorials = internal::factorial<_degree>();
        MultiIndices multi_indices;
        internal::generate_multi_indices(_simplex_dim, _degree, multi_indices);

        Point result = Point::Zero(_ordinate_dim);
        for (size_t i = 0; i < m_num_control_points; i++) {
            std::span<uint8_t, _simplex_dim + 1> multi_index(
                multi_indices.data() + i * (_simplex_dim + 1), _simplex_dim + 1);
            Scalar coeff = factorials[_degree];
            for (size_t j = 0; j < _simplex_dim; j++) {
                coeff /= factorials[multi_index[j]];
                coeff *= std::pow(b[static_cast<Eigen::Index>(j)], multi_index[j]);
            }
            result += coeff * m_ordinates.row(static_cast<Eigen::Index>(i));
        }
        return result;
    }

    Point evaluate_directional_derivative(BarycentricPoint b, BarycentricPoint dir) const
    {
        return Point::Zero(_ordinate_dim);
    }

protected:
    Ordinates m_ordinates;
};

} // namespace nanospline
