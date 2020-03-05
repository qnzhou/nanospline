#pragma once

#include <cassert>
#include <vector>

#include <Eigen/Eigenvalues>

#include <nanospline/Exceptions.h>

namespace nanospline
{
/**
 * Compute the roots for a real polynomial of degree _degree.
 * The coeffs are assumed to be from zero to degree.
 */
template <typename Scalar, int _degree>
class PolynomialRootFinder
{
    public:
    static void find_real_roots_in_interval(const std::vector<Scalar> &coeffs, std::vector<Scalar> &roots, const Scalar t0, const Scalar t1, const Scalar eps)
    {
        //This is for efficienty and compilation time, the algorithm works no matter
        static_assert(_degree <= 16, "This is for efficienty and compilation time, the algorithm works no matter");

        using std::abs;
        assert(coeffs.size() > _degree);
        assert(t0 < t1);

        //Largest degree is zero, the polynomial is one degree less
        if (abs(coeffs[_degree]) < eps)
        {
            PolynomialRootFinder<Scalar, _degree-1>::find_real_roots_in_interval(coeffs, roots, t0, t1, eps);
            return;
        }

        typedef Eigen::Matrix<Scalar, _degree, _degree> MatType;

        MatType companion;
        companion.setZero();
        for (int i = 0; i < _degree - 1; ++i)
            companion(i + 1, i) = 1;

        for (int i = 0; i < _degree; ++i)
            companion(i, _degree - 1) =
                -coeffs[static_cast<size_t>(i)] / coeffs[_degree];

        Eigen::EigenSolver<MatType> es(companion, false);
        const auto &vals = es.eigenvalues();

        for (int i = 0; i < vals.size(); ++i)
        {
            const auto lambda = vals(i);
            const Scalar current_t = lambda.real();

            if (abs(abs(lambda) - abs(current_t)) > eps)
                continue;

            if (current_t >= t0 && current_t <= t1)
                roots.push_back(current_t);
        }
    }
};

template <typename Scalar>
class PolynomialRootFinder<Scalar, 2>
{
    public:
    static void find_real_roots_in_interval(const std::vector<Scalar> &coeffs, std::vector<Scalar> &roots, const Scalar t0, const Scalar t1, const Scalar eps)
    {
        using std::abs;
        assert(coeffs.size() > 2);

        //Largest degree is zero, the polynomial is one degree less
        if (abs(coeffs[2]) < eps)
        {
            PolynomialRootFinder<Scalar, 1>::find_real_roots_in_interval(coeffs, roots, t0, t1, eps);
            return;
        }

        const Scalar a = coeffs[2];
        const Scalar b = coeffs[1];
        const Scalar c = coeffs[0];

        const Scalar discr = b * b - 4 * a * c;
        //no real root
        if (discr < 0)
            return;

        //dublicate root
        if (abs(discr) < eps)
        {
            const Scalar root = -b / (2 * a);
            if (root >= t0 && root <= t1)
                roots.push_back(root);
            return;
        }

        const Scalar sqrt_discr = std::sqrt(discr);
        Scalar root = (-b - sqrt_discr) / (2 * a);
        if (root >= t0 && root <= t1)
            roots.push_back(root);

        root = (-b + sqrt_discr) / (2 * a);
        if (root >= t0 && root <= t1)
            roots.push_back(root);
    }
};

template <typename Scalar>
class PolynomialRootFinder<Scalar, 1>
{
    public:
    static void find_real_roots_in_interval(const std::vector<Scalar> &coeffs, std::vector<Scalar> &roots, const Scalar t0, const Scalar t1, const Scalar eps)
    {
        using std::abs;
        assert(coeffs.size() > 1);

        //Largest degree is zero, the polynomial is one degree less
        if (abs(coeffs[1]) < eps)
        {
            PolynomialRootFinder<Scalar, 0>::find_real_roots_in_interval(coeffs, roots, t0, t1, eps);
            return;
        }

        const Scalar a = coeffs[1];
        const Scalar b = coeffs[0];

        const Scalar root = -b / a;
        if (root >= t0 && root <= t1)
                roots.push_back(root);
    }
};

template <typename Scalar>
class PolynomialRootFinder<Scalar, 0> {
    public:
    static void find_real_roots_in_interval(const std::vector<Scalar> &coeffs, std::vector<Scalar> &roots, const Scalar t0, const Scalar t1, const Scalar eps)
    {
        using std::abs;
        assert(coeffs.size() > 0);

        if (abs(coeffs[0]) < eps)
            throw invalid_setting_error("Polynomial is zero = zero, has infinit roots");
    }
};

} // namespace nanospline
