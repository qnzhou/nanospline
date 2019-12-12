#pragma once

#include <nanospline/auto_optimal_turning_angle.h>
#include <nanospline/conversion.h>
#include <nanospline/Bezier.h>
#include <nanospline/RationalBezier.h>

#include <vector>

namespace nanospline
{
    template <typename Curve>
    class OptimalPointsToReduceTurningAngle {
        public:
        static std::vector<typename Curve::Scalar> compute(const Curve &curve, bool flip, typename Curve::Scalar t0 = 0, typename Curve::Scalar t1 = 1)
        {
            throw invalid_setting_error("Optimal turning angle works only on 2D curve, try changing the template argument.");
        }
    };


    template <typename Scalar, int degree, bool generic >
    class OptimalPointsToReduceTurningAngle<Bezier<Scalar, 2, degree, generic>>
    {
    public:
        static std::vector<Scalar> compute(const Bezier<Scalar, 2, degree, generic> &curve, bool flip, Scalar t0 = 0, Scalar t1 = 1){
            auto res = optimal_points_to_reduce_turning_angle(curve, flip, t0, t1);

            std::sort(res.begin(), res.end());
            res.erase(std::unique(res.begin(), res.end()), res.end());

            return res;
        }
    };

    template <typename Scalar, int degree, bool generic>
    class OptimalPointsToReduceTurningAngle<RationalBezier<Scalar, 2, degree, generic>>
    {
    public:
        static std::vector<Scalar> compute(const RationalBezier<Scalar, 2, degree, generic> &curve, bool flip, Scalar t0 = 0, Scalar t1 = 1)
        {
            auto res = optimal_points_to_reduce_turning_angle(curve, flip, t0, t1);

            std::sort(res.begin(), res.end());
            res.erase(std::unique(res.begin(), res.end()), res.end());

            return res;
        }
    };

    template <typename Scalar, int degree, bool generic>
    class OptimalPointsToReduceTurningAngle<BSpline<Scalar, 2, degree, generic>>
    {
    public:
        static std::vector<Scalar> compute(const BSpline<Scalar, 2, degree, generic> &curve, bool flip, Scalar t0 = 0, Scalar t1 = 1)
        {
            throw not_implemented_error("Missing convertion from nurbs to rational bezier");
            //TODO map intervals
            Scalar res;
            // const auto beziers = convert_to_Bezier(curve);
            // for(const auto &bezier : beziers){
            //     auto tmp = optimal_points_to_reduce_turning_angle(curve, flip, t0, t1);
            //     res.insert(res.end(), tmp.begin(), tmp.end());
            // }

            std::sort(res.begin(), res.end());
            res.erase(std::unique(res.begin(), res.end()), res.end());

            return res;
        }
    };

    template <typename Scalar, int degree, bool generic>
    class OptimalPointsToReduceTurningAngle<NURBS<Scalar, 2, degree, generic>>
    {
    public:
        static std::vector<Scalar> compute(const NURBS<Scalar, 2, degree, generic> &curve, bool flip, Scalar t0 = 0, Scalar t1 = 1)
        {
            //TODO map intervals
            Scalar res;
            const auto beziers = convert_to_RationalBezier(curve);
            for (const auto &bezier : beziers)
            {
                auto tmp = optimal_points_to_reduce_turning_angle(curve, flip, t0, t1);
                res.insert(res.end(), tmp.begin(), tmp.end());
            }

            std::sort(res.begin(), res.end());
            res.erase(std::unique(res.begin(), res.end()), res.end());

            return res;
        }
    };
}
