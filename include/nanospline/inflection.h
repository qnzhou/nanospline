#pragma once

#include <nanospline/auto_inflection.h>
#include <nanospline/conversion.h>
#include <nanospline/Bezier.h>
#include <nanospline/RationalBezier.h>

#include <vector>

namespace nanospline
{
    template <typename Curve>
    class InlfectionPoints {
        public:
        static std::vector<typename Curve::Scalar> compute(const Curve &curve, typename Curve::Scalar t0 = 0, typename Curve::Scalar t1 = 1)
        {
            throw invalid_setting_error("Inflection works only on 2D curve, try changing the template argument.");
        }
    };


    template <typename Scalar, int degree, bool generic >
    class InlfectionPoints<Bezier<Scalar, 2, degree, generic>>
    {
    public:
        static std::vector<Scalar> compute(const Bezier<Scalar, 2, degree, generic> &curve, Scalar t0 = 0, Scalar t1 = 1){
            auto res = compute_inflections(curve, t0, t1);

            std::sort(res.begin(), res.end());
            res.erase(std::unique(res.begin(), res.end()), res.end());

            return res;
        }
    };

    template <typename Scalar, int degree, bool generic>
    class InlfectionPoints<RationalBezier<Scalar, 2, degree, generic>>
    {
    public:
        static std::vector<Scalar> compute(const RationalBezier<Scalar, 2, degree, generic> &curve, Scalar t0 = 0, Scalar t1 = 1)
        {
            auto res = compute_inflections(curve, t0, t1);

            std::sort(res.begin(), res.end());
            res.erase(std::unique(res.begin(), res.end()), res.end());

            return res;
        }
    };

    template <typename Scalar, int degree, bool generic>
    class InlfectionPoints<BSpline<Scalar, 2, degree, generic>>
    {
    public:
        static std::vector<Scalar> compute(const BSpline<Scalar, 2, degree, generic> &curve, Scalar t0 = 0, Scalar t1 = 1)
        {
            //TODO map intervals
            std::vector<Scalar> res;
            const auto beziers = convert_to_Bezier(curve);
            for(const auto &bezier : beziers){
                auto tmp = compute_inflections(curve, t0, t1);
                res.insert(res.end(), tmp.begin(), tmp.end());
            }

            std::sort(res.begin(), res.end());
            res.erase(std::unique(res.begin(), res.end()), res.end());

            return res;
        }
    };

    template <typename Scalar, int degree, bool generic>
    class InlfectionPoints<NURBS<Scalar, 2, degree, generic>>
    {
    public:
        static std::vector<Scalar> compute(const NURBS<Scalar, 2, degree, generic> &curve, Scalar t0 = 0, Scalar t1 = 1)
        {
            throw not_implemented_error("Missing convertion from nurbs to rational bezier");
            //TODO map intervals
            std::vector<Scalar> res;
            // const auto beziers = convert_to_RationalBezier(curve);
            // for (const auto &bezier : beziers)
            // {
            //     auto tmp = compute_inflections(curve, t0, t1);
            //     res.insert(res.end(), tmp.begin(), tmp.end());
            // }

            // std::sort(res.begin(), res.end());
            // res.erase(std::unique(res.begin(), res.end()), res.end());

            return res;
        }
    };
}
