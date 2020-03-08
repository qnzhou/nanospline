#pragma once

#include <cassert>

#include <Eigen/Core>

namespace nanospline {

template<typename _Scalar, int _dim>
class PatchBase {
    public:
        using Scalar = _Scalar;
        using Point = Eigen::Matrix<Scalar, 1, _dim>;

    public:
        virtual ~PatchBase() = default;
        virtual Point evaluate(Scalar u, Scalar v) const =0;
        virtual Point evaluate_derivative_u(Scalar u, Scalar v) const=0;
        virtual Point evaluate_derivative_v(Scalar u, Scalar v) const=0;
};

}
