#pragma once
#include <nanospline/Exceptions.h>

#include <Eigen/Core>


namespace nanospline {

/**
 * Base class for all simplex types.
 *
 * @tparam _Scalar       The scalar type.
 * @tparam _simplex_dim  The dimension of the simplex.
 * @tparam _ordinate_dim The dimension of the ordinate space.
 *                       If -1, the ordinate dimension is dynamic.
 */
template <typename _Scalar, uint8_t _simplex_dim, int8_t _ordinate_dim = -1>
class SimplexBase
{
public:
    static_assert(_simplex_dim >= 0, "Negative dimension is not allowed");
    using Scalar = _Scalar;
    using BarycentricPoint = Eigen::Matrix<Scalar, 1, _simplex_dim>;
    using Point = Eigen::Matrix<Scalar, 1, _ordinate_dim>;

public:
    virtual ~SimplexBase() = default;
    virtual void initialize() {}

    constexpr uint8_t get_simplex_dim() const { return _simplex_dim; }
    constexpr int8_t get_ordinate_dim() const { return _ordinate_dim; }

    virtual Point evaluate(BarycentricPoint b) const = 0;
    virtual Point evaluate_directional_derivative(
        BarycentricPoint b, BarycentricPoint dir) const = 0;
};

} // namespace nanospline
