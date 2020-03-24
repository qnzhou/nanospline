#pragma once

#include <nanospline/Exceptions.h>

namespace nanospline {

template<int N>
class Quadrature {
    public:
        template<typename FunctionType, typename Scalar>
        static Scalar integrate(FunctionType& func,
                Scalar t0, Scalar t1) {
            throw not_implemented_error(
                    "Please specify the number of quadrature points");
        }
};

template<>
class Quadrature<2> {
    public:
        template<typename FunctionType, typename Scalar>
        static Scalar integrate(FunctionType& func,
                Scalar t0, Scalar t1) {
            const Scalar q0 = t0 + (t1-t0)/2*(-1.0/std::sqrt(3) + 1);
            const Scalar q1 = t0 + (t1-t0)/2*( 1.0/std::sqrt(3) + 1);

            const auto val0 = func(q0);
            const auto val1 = func(q1);
            return (val0 + val1) * (t1-t0) * 0.5;
        }
};

template<>
class Quadrature<3> {
    public:
        template<typename FunctionType, typename Scalar>
        static Scalar integrate(FunctionType& func,
                Scalar t0, Scalar t1) {
            const Scalar q0 = 0.5 * (t0 + t1);
            const Scalar q1 = t0 + (t1-t0)/2*(-std::sqrt(0.6) + 1);
            const Scalar q2 = t0 + (t1-t0)/2*( std::sqrt(0.6) + 1);

            const auto val0 = func(q0);
            const auto val1 = func(q1);
            const auto val2 = func(q2);

            constexpr Scalar w0 = 8.0/9.0;
            constexpr Scalar w1 = 5.0/9.0;
            constexpr Scalar w2 = 5.0/9.0;

            return (val0*w0 + val1*w1 + val2*w2) * (t1-t0) * 0.5;
        }
};

template<>
class Quadrature<4> {
    public:
        template<typename FunctionType, typename Scalar>
        static Scalar integrate(FunctionType& func,
                Scalar t0, Scalar t1) {
            const Scalar r0 = std::sqrt(3.0/7.0 - 2.0/7.0*std::sqrt(1.2));
            const Scalar r1 = std::sqrt(3.0/7.0 + 2.0/7.0*std::sqrt(1.2));

            const Scalar q0 = t0 + (t1-t0)/2*(-r0+1);
            const Scalar q1 = t0 + (t1-t0)/2*( r0+1);
            const Scalar q2 = t0 + (t1-t0)/2*(-r1+1);
            const Scalar q3 = t0 + (t1-t0)/2*( r1+1);

            const auto val0 = func(q0);
            const auto val1 = func(q1);
            const auto val2 = func(q2);
            const auto val3 = func(q3);

            const Scalar w0 = (18 + std::sqrt(30))/36;
            const Scalar w1 = (18 - std::sqrt(30))/36;

            return (val0*w0 + val1*w0 +
                    val2*w1 + val3*w1) * (t1-t0) * 0.5;
        }
};

template<>
class Quadrature<5> {
    public:
        template<typename FunctionType, typename Scalar>
        static Scalar integrate(FunctionType& func,
                Scalar t0, Scalar t1) {
            const Scalar r0 = sqrt(5-2*sqrt(10.0/7.0)) / 3;
            const Scalar r1 = sqrt(5+2*sqrt(10.0/7.0)) / 3;

            const Scalar q0 = 0.5 * (t0 + t1);
            const Scalar q1 = t0 + (t1-t0)/2*(-r0+1);
            const Scalar q2 = t0 + (t1-t0)/2*( r0+1);
            const Scalar q3 = t0 + (t1-t0)/2*(-r1+1);
            const Scalar q4 = t0 + (t1-t0)/2*( r1+1);

            const auto val0 = func(q0);
            const auto val1 = func(q1);
            const auto val2 = func(q2);
            const auto val3 = func(q3);
            const auto val4 = func(q4);

            const Scalar w0 = 128.0/225.0;
            const Scalar w1 = (322 + 13*std::sqrt(70))/900;
            const Scalar w2 = (322 - 13*std::sqrt(70))/900;

            return (val0*w0 +
                    val1*w1 + val2*w1 +
                    val3*w2 + val4*w2) * (t1-t0) * 0.5;
        }
};


}
