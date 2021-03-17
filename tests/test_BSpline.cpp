#include <catch2/catch.hpp>

#include <nanospline/BSpline.h>
#include <nanospline/Bezier.h>
#include <nanospline/forward_declaration.h>
#include <nanospline/save_msh.h>

#include "validation_utils.h"

TEST_CASE("BSpline", "[nonrational][bspline]")
{
    using namespace nanospline;
    using Scalar = double;

    SECTION("Generic degree 0")
    {
        Eigen::Matrix<Scalar, 3, 2> control_pts;
        control_pts << 0.0, 0.0, 1.0, 0.0, 2.0, 0.0;
        Eigen::Matrix<Scalar, 4, 1> knots;
        knots << 0.0, 0.5, 0.75, 1.0;

        BSpline<Scalar, 2, 0, true> curve;
        curve.set_control_points(control_pts);
        curve.set_knots(knots);

        REQUIRE(curve.get_degree() == 0);

        auto p0 = curve.evaluate(0.1);
        REQUIRE(p0[0] == Approx(0.0));
        REQUIRE(p0[1] == Approx(0.0));

        auto p1 = curve.evaluate(0.6);
        REQUIRE(p1[0] == Approx(1.0));
        REQUIRE(p1[1] == Approx(0.0));

        auto p2 = curve.evaluate(1.0);
        REQUIRE(p2[0] == Approx(2.0));
        REQUIRE(p2[1] == Approx(0.0));

        SECTION("Derivative")
        {
            validate_derivatives(curve, 10);
            validate_2nd_derivatives(curve, 10);
        }

        SECTION("Degree elevation") { REQUIRE_THROWS(curve.elevate_degree()); }

        SECTION("Out of bound evaluation should extrapolate")
        {
            auto p3 = curve.evaluate(-0.1);
            REQUIRE(p3[0] == Approx(0.0));
            REQUIRE(p3[1] == Approx(0.0));
            auto p4 = curve.evaluate(1.1);
            REQUIRE(p4[0] == Approx(2.0));
            REQUIRE(p4[1] == Approx(0.0));
        }

        SECTION("Update") { offset_and_validate(curve); }
    }

    SECTION("Generic degree 1")
    {
        Eigen::Matrix<Scalar, 4, 2> control_pts;
        control_pts << 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0;
        Eigen::Matrix<Scalar, 6, 1> knots;
        knots << 0.0, 0.0, 0.2, 0.8, 1.0, 1.0;

        BSpline<Scalar, 2, 1, true> curve;
        curve.set_control_points(control_pts);
        curve.set_knots(knots);

        auto p0 = curve.evaluate(0.1);
        REQUIRE(p0[0] == Approx(0.5));
        REQUIRE(p0[1] == Approx(0.0));

        auto p1 = curve.evaluate(0.5);
        REQUIRE(p1[0] == Approx(1.5));
        REQUIRE(p1[1] == Approx(0.0));

        auto p2 = curve.evaluate(0.9);
        REQUIRE(p2[0] == Approx(2.5));
        REQUIRE(p2[1] == Approx(0.0));

        SECTION("Derivative")
        {
            validate_derivatives(curve, 10);
            validate_2nd_derivatives(curve, 10);
        }

        SECTION("Knot insertion and removal")
        {
            auto curve2 = curve;
            curve2.insert_knot(0.5, 1);
            assert_same(curve, curve2, 10);

            REQUIRE(curve2.remove_knot(0.5, 1) == 1);
            assert_same(curve, curve2, 10);

            REQUIRE(curve2.remove_knot(0.5, 1) == 0);
            assert_same(curve, curve2, 10);
        }

        SECTION("Turning angle")
        {
#if NANOSPLINE_SYMPY
            auto total_turning_angle = curve.get_turning_angle(0, 1);
            REQUIRE(total_turning_angle == Approx(0));
            const auto split_pts = curve.reduce_turning_angle(0, 1);
            REQUIRE(split_pts.size() == 0);
#endif
        }

        SECTION("Split and combine")
        {
            const auto r = curve.convert_to_Bezier();
            decltype(curve) curve2(std::get<0>(r), std::get<1>(r));
            assert_same(curve, curve2, 10);
        }

        SECTION("Degree elevation")
        {
            const auto curve2 = curve.elevate_degree();
            assert_same(curve, curve2, 10);
        }

        SECTION("Out of bound evaluation should extrapolate")
        {
            auto p3 = curve.evaluate(-0.2);
            REQUIRE(p3[0] == Approx(-1.0));
            REQUIRE(p3[1] == Approx(0.0));
            auto p4 = curve.evaluate(1.2);
            REQUIRE(p4[0] == Approx(4.0));
            REQUIRE(p4[1] == Approx(0.0));
        }

        SECTION("Update") { offset_and_validate(curve); }
    }

    SECTION("Generic degree 2")
    {
        Eigen::Matrix<Scalar, 4, 2> control_pts;
        control_pts << 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0;
        Eigen::Matrix<Scalar, 7, 1> knots;
        knots << 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0;

        BSpline<Scalar, 2, 2, true> curve;
        curve.set_control_points(control_pts);
        curve.set_knots(knots);

        auto p0 = curve.evaluate(0.0);
        REQUIRE(p0[0] == Approx(0.0));
        REQUIRE(p0[1] == Approx(0.0));

        auto p1 = curve.evaluate(0.5);
        REQUIRE(p1[0] == Approx(1.5));
        REQUIRE(p1[1] == Approx(0.0));

        auto p2 = curve.evaluate(1.0);
        REQUIRE(p2[0] == Approx(3.0));
        REQUIRE(p2[1] == Approx(0.0));

        auto p3 = curve.evaluate(0.1);
        auto p4 = curve.evaluate(0.9);
        REQUIRE(p3[0] == Approx(3.0 - p4[0]));
        REQUIRE(p3[1] == Approx(0.0));
        REQUIRE(p4[1] == Approx(0.0));

        SECTION("Derivative")
        {
            validate_derivatives(curve, 10);
            validate_2nd_derivatives(curve, 10);
        }

        SECTION("Knot insertion and removal")
        {
            auto curve2 = curve;
            curve2.insert_knot(0.1, 2);
            assert_same(curve, curve2, 10);
            REQUIRE(curve2.remove_knot(0.1, 1) == 1);
            assert_same(curve, curve2, 10);
            REQUIRE(curve2.remove_knot(0.1, 1) == 1);
            assert_same(curve, curve2, 10);
        }

        SECTION("Turning angle")
        {
#if NANOSPLINE_SYMPY
            auto total_turning_angle = curve.get_turning_angle(0, 1);
            REQUIRE(total_turning_angle == Approx(0));
            const auto split_pts = curve.reduce_turning_angle(0, 1);
            REQUIRE(split_pts.size() == 0);
#endif
        }

        SECTION("Singularity")
        {
#if NANOSPLINE_SYMPY
            auto singular_pts = curve.compute_singularities(0, 1);
            REQUIRE(singular_pts.size() == 0);
#endif
        }

        SECTION("Split and combine")
        {
            const auto r = curve.convert_to_Bezier();
            decltype(curve) curve2(std::get<0>(r), std::get<1>(r));
            assert_same(curve, curve2, 10);
        }

        SECTION("Degree elevation")
        {
            const auto curve2 = curve.elevate_degree();
            assert_same(curve, curve2, 10);
        }

        SECTION("Out of bound evaluation should extrapolate")
        {
            auto p3 = curve.evaluate(-1.0);
            REQUIRE(p3[0] < 0.0);
            REQUIRE(p3[1] == Approx(0.0));
            auto p4 = curve.evaluate(2.0);
            REQUIRE(p4[0] > 3.0);
            REQUIRE(p4[1] == Approx(0.0));
        }

        SECTION("Update") { offset_and_validate(curve); }
    }

    SECTION("Generic degree 3")
    {
        Eigen::Matrix<Scalar, 4, 2> control_pts;
        control_pts << 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0;
        Eigen::Matrix<Scalar, 8, 1> knots;
        knots << 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0;

        BSpline<Scalar, 2, 3, true> curve;
        curve.set_control_points(control_pts);
        curve.set_knots(knots);

        // B-spline without internal knots is a Bezier curve.
        Bezier<Scalar, 2, 3> bezier_curve;
        bezier_curve.set_control_points(control_pts);
        assert_same(curve, bezier_curve, 10);

        SECTION("Derivative")
        {
            validate_derivatives(curve, 10);
            validate_2nd_derivatives(curve, 10);
        }

        SECTION("Knot insertion")
        {
            auto curve2 = curve;
            curve2.insert_knot(0.5);
            assert_same(curve, curve2, 10);

            curve2.insert_knot(0.5);
            assert_same(curve, curve2, 10);

            curve2.insert_knot(0.5, 1);
            assert_same(curve, curve2, 10);

            curve2.insert_knot(0.6, 3);
            assert_same(curve, curve2, 10);

            REQUIRE(curve2.remove_knot(0.6, 2) == 2);
            assert_same(curve, curve2, 10);

            REQUIRE(curve2.remove_knot(0.6, 1) == 1);
            assert_same(curve, curve2, 10);

            REQUIRE(curve2.remove_knot(0.5, 3) == 3);
            assert_same(curve, curve2, 10);
        }

        SECTION("Turning angle")
        {
#if NANOSPLINE_SYMPY
            auto total_turning_angle = curve.get_turning_angle(0, 1);
            REQUIRE(total_turning_angle == Approx(0));
            const auto split_pts = curve.reduce_turning_angle(0, 1);
            REQUIRE(split_pts.size() == 0);
#endif
        }

        SECTION("Singularity")
        {
#if NANOSPLINE_SYMPY
            auto singular_pts = curve.compute_singularities(0, 1);
            REQUIRE(singular_pts.size() == 0);
#endif
        }

        SECTION("Split and combine")
        {
            const auto r = curve.convert_to_Bezier();
            decltype(curve) curve2(std::get<0>(r), std::get<1>(r));
            assert_same(curve, curve2, 10);
        }

        SECTION("Degree elevation")
        {
            const auto curve2 = curve.elevate_degree();
            assert_same(curve, curve2, 10);
        }

        SECTION("Out of bound evaluation should extrapolate")
        {
            auto p3 = curve.evaluate(-1.0);
            REQUIRE(p3[0] < 0.0);
            REQUIRE(p3[1] == Approx(0.0));
            auto p4 = curve.evaluate(2.0);
            REQUIRE(p4[0] > 3.0);
            REQUIRE(p4[1] == Approx(0.0));
        }

        SECTION("Update") { offset_and_validate(curve); }
    }

    SECTION("Approximate closest point")
    {
        Eigen::Matrix<Scalar, 10, 2> control_pts;
        control_pts << 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0, 7.0,
            0.0, 8.0, 0.0, 9.0, 0.0;
        Eigen::Matrix<Scalar, 14, 1> knots;
        knots << 0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.9, 1.0, 1.0, 1.0, 1.0;

        BSpline<Scalar, 2, 3, true> curve;
        curve.set_control_points(control_pts);
        curve.set_knots(knots);

        validate_approximate_inverse_evaluation(curve, 10);
    }

    SECTION("Closed BSpline")
    {
        Eigen::Matrix<Scalar, Eigen::Dynamic, 3> control_pts(98, 3);
        control_pts << 116.77856840140299, 0.990973072246042, 398.531783226971, 116.77856840140299,
            -0.9909730722460299, 398.531783226971, 116.74214258712, -1.96538630167913,
            398.621940366223, 116.63509969331801, -3.4031822484227403, 398.88688107666997,
            116.59061975410499, -3.87844738884722, 398.996972893936, 116.48366040341101,
            -4.820794779603521, 399.26170682832503, 116.42087907691901, -5.2891556063723995,
            399.41709621204103, 116.209347750077, -6.6515598956308395, 399.940655117481,
            116.036754359033, -7.51312136271168, 400.367839159197, 115.73092979908401,
            -8.73602915819255, 401.12478223543604, 115.620701643976, -9.1333236747386,
            401.397606756229, 115.385687684641, -9.897811751176421, 401.979287280405,
            115.26203119570599, -10.2616150505271, 402.285348127648, 114.87318930320501,
            -11.299430307622401, 403.247766521579, 114.59030066243001, -11.920354975591799,
            403.947941163824, 114.12962960386699, -12.742773704811102, 405.088143172522,
            113.970017896896, -12.9981196093857, 405.483196402463, 113.645611156587,
            -13.4608944017134, 406.286132061621, 113.314320700286, -13.8789656101838,
            407.106105536231, 112.96947083813801, -14.2089805922911, 407.95963975948,
            112.617748966644, -14.4943680123967, 408.830182824222, 112.437485302958,
            -14.6152145646128, 409.276351503846, 111.893493639978, -14.9041921303441,
            410.62277949876204, 111.52891094374901, -14.999478460405099, 411.525154272187,
            110.97923144310599, -15.0002575979545, 412.88566020198505, 110.79553400089499,
            -14.976655034467399, 413.340327804551, 110.427168588273, -14.880273975166501,
            414.252065158733, 110.243631670512, -14.8075435699898, 414.706335453868,
            109.703129748937, -14.5206022870023, 416.04412608332905, 109.35285911324601,
            -14.237518726347599, 416.911077264571, 108.84263318388, -13.674442679786099,
            418.17393213041396, 108.674218098707, -13.4621174207831, 418.590774549993,
            108.34864087302499, -12.9977392080388, 419.396607347268, 108.191184363668,
            -12.7457005272554, 419.786326314189, 107.734427070851, -11.931465539438701,
            420.916841539278, 107.45058613295, -11.3117175810457, 421.6193732999, 106.92584403691,
            -9.91237172626739, 422.918157027883, 106.69278670393, -9.15316796691815,
            423.49499482478, 106.38747009468, -7.9349668239793605, 424.250680814139,
            106.29300396347601, -7.51523002917608, 424.484492961649, 106.118991283651,
            -6.64784172308645, 424.915189952748, 106.039216289581, -6.19819024187188,
            425.11264021932, 105.827523422949, -4.83487351323821, 425.636599055351,
            105.721216323796, -3.89987336047708, 425.899718664195, 105.613341742281,
            -2.45867435863919, 426.16671793285997, 105.586006063323, -1.9707862816348,
            426.234376191158, 105.549461881335, -0.9857166368346689, 426.324826320796,
            105.54044589909701, -0.494750789328092, 426.347141685899, 105.54025146589301,
            0.9735948956328051, 426.347622925528, 105.575673913195, 1.94646569556888,
            426.25994918972697, 105.68321983880399, 3.39654863941427, 425.993763373907,
            105.72838501664799, 3.87886550363307, 425.88197550624, 105.835093979257,
            4.81915162597785, 425.61786124974196, 105.958504358918, 5.74054684133611,
            425.312409488061, 106.114756472455, 6.624794901337809, 424.925671490323,
            106.287624528891, 7.49024873427974, 424.49780754474, 106.382849745285,
            7.915467430503489, 424.262116593276, 106.691002866419, 9.14812322500516,
            423.49940998255903, 106.923525476506, 9.90411438109418, 422.92389567280605,
            107.311718825646, 10.9423980465413, 421.963082334131, 107.447732466367,
            11.2724264516109, 421.626436381759, 107.733369874828, 11.8994320157942, 420.91945819606,
            107.882583758245, 12.1951185748258, 420.550140463204, 108.340888651724,
            13.013468636116901, 419.415794787727, 108.663480552166, 13.474751870911199,
            418.617350937168, 109.17300532879801, 14.0399848183879, 417.35623148688103,
            109.348157453899, 14.2078656236473, 416.922714294547, 109.70161128489201,
            14.4947454890836, 416.04788442018895, 109.87978045866099, 14.6140065653552,
            415.606899767003, 110.418503492242, 14.9011184973099, 414.27351204442397,
            110.78323203875901, 14.998982480015599, 413.37077625893596, 111.338990832463,
            15.0004970353436, 411.995223534819, 111.52393256346501, 14.9764362091141,
            411.537476210994, 111.88937485667701, 14.8809861560943, 410.632973857962,
            112.070488342957, 14.8096047110738, 410.18470178685703, 112.42953104164299,
            14.6199858328352, 409.296039011608, 112.607460326034, 14.501749458884, 408.855648129292,
            112.96007340349901, 14.217519921627499, 407.982899249968, 113.13559861266701,
            14.0504296267554, 407.54845867296604, 113.646880347447, 13.4862544146429,
            406.28299069884997, 113.969823791972, 13.026013372250299, 405.48367682801603,
            114.42779616489, 12.2114558366078, 404.350154307124, 114.576047331389, 11.9190214593258,
            403.983219432426, 114.863222432949, 11.291228473699, 403.27243541626603,
            114.99979588004899, 10.9610222424249, 402.934403942369, 115.389203346499,
            9.92317047569851, 401.970585702383, 115.621914451387, 9.16873357797236,
            401.394604949059, 115.929962161262, 7.94169779470261, 400.632159378298,
            116.025922694604, 7.51530625740308, 400.394648495839, 116.19980096132599,
            6.64865840298916, 399.964284271649, 116.27829874762901, 6.20635433084811,
            399.76999524716797, 116.418724412343, 5.3041186477659, 399.42242919908597,
            116.480651946936, 4.8441866710579395, 399.26915302647296, 116.58776844069101,
            3.90651574565541, 399.004030148977, 116.633060972542, 3.42736375284877,
            398.891927092441, 116.742138394356, 1.97167065300644, 398.621950743678,
            116.77856840140299, 0.990973072246042, 398.531783226971, 116.77856840140299,
            -0.9909730722460299, 398.531783226971;

        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> knots(102, 1);
        knots << -0.0029307229988643074, -0.0029307229988643074, 0.0, 0.0, 0.00293072299886427,
            0.00293072299886427, 0.0043960844982964, 0.0043960844982964, 0.00586144599772854,
            0.00586144599772854, 0.00879216899659286, 0.00879216899659286, 0.010257530496025,
            0.010257530496025, 0.0117228919954572, 0.0117228919954572, 0.0146536149943214,
            0.0146536149943214, 0.0161189764937534, 0.0161189764937534, 0.0175843379931855,
            0.0190496994926175, 0.0190496994926175, 0.0205150609920496, 0.0205150609920496,
            0.0234457839909138, 0.0234457839909138, 0.0249111454903459, 0.0249111454903459,
            0.0263765069897781, 0.0263765069897781, 0.0293072299886423, 0.0293072299886423,
            0.0307725914880745, 0.0307725914880745, 0.0322379529875068, 0.0322379529875068,
            0.0351686759863713, 0.0351686759863713, 0.0380993989852358, 0.0380993989852358,
            0.0395647604846681, 0.0395647604846681, 0.0410301219841004, 0.0410301219841004,
            0.0439608449829649, 0.0439608449829649, 0.0454262064823972, 0.0454262064823972,
            0.0468915679818295, 0.0468915679818295, 0.0498222909806941, 0.0498222909806941,
            0.0512876524801264, 0.0512876524801264, 0.0527530139795587, 0.054218375478991,
            0.054218375478991, 0.0556837369784233, 0.0556837369784233, 0.058614459977288,
            0.058614459977288, 0.0600798214767204, 0.0600798214767204, 0.0615451829761528,
            0.0615451829761528, 0.0644759059750176, 0.0644759059750176, 0.0659412674744499,
            0.0659412674744499, 0.0674066289738822, 0.0674066289738822, 0.0703373519727467,
            0.0703373519727467, 0.0718027134721789, 0.0718027134721789, 0.0732680749716111,
            0.0732680749716111, 0.0747334364710432, 0.0747334364710432, 0.0761987979704754,
            0.0761987979704754, 0.0791295209693398, 0.0791295209693398, 0.0805948824687719,
            0.0805948824687719, 0.0820602439682041, 0.0820602439682041, 0.0849909669670685,
            0.0849909669670685, 0.0864563284665006, 0.0864563284665006, 0.0879216899659328,
            0.0879216899659328, 0.0893870514653649, 0.0893870514653649, 0.0908524129647971,
            0.0908524129647971, 0.0937831359636614, 0.0937831359636614, 0.09671385896252567,
            0.09671385896252567;

        BSpline<Scalar, 3, 3, true> curve;
        curve.set_control_points(control_pts);
        curve.set_knots(knots);

        const Scalar min_t = curve.get_domain_lower_bound();
        const Scalar max_t = curve.get_domain_upper_bound();
        constexpr int N = 100;
        for (int i = 0; i < N + 1; i++) {
            Scalar t = min_t + Scalar(i) / Scalar(N) * (max_t - min_t);
            t = std::min(max_t, t);
            REQUIRE_NOTHROW(curve.evaluate(t));
        }

        // Check curve is indeed closed.
        const auto min_p = curve.evaluate(min_t);
        const auto max_p = curve.evaluate(max_t);
        REQUIRE((max_p - min_p).norm() == Approx(0.0).margin(1e-6));

        SECTION("Derivative")
        {
            validate_derivatives(curve, 10);
            validate_2nd_derivatives(curve, 10);
        }

        SECTION("Split and combine")
        {
            const auto r = curve.convert_to_Bezier();
            decltype(curve) curve2(std::get<0>(r), std::get<1>(r));
            assert_same(curve, curve2, 10);
        }

        SECTION("Degree elevation")
        {
            const auto curve2 = curve.elevate_degree();
            assert_same(curve, curve2, 10);
        }

        SECTION("Approximate inverse evaluate")
        {
            validate_approximate_inverse_evaluation(curve, 10);
        }

        SECTION("Update") { offset_and_validate(curve); }
    }

    SECTION("Comparison")
    {
        SECTION("Open curve")
        {
            Eigen::Matrix<Scalar, 10, 2> ctrl_pts;
            ctrl_pts << 1, 4, .5, 6, 5, 4, 3, 12, 11, 14, 8, 4, 12, 3, 11, 9, 15, 10, 17, 8;
            Eigen::Matrix<Scalar, 14, 1> knots;
            knots << 0, 0, 0, 0, 1.0 / 7, 2.0 / 7, 3.0 / 7, 4.0 / 7, 5.0 / 7, 6.0 / 7, 1, 1, 1, 1;

            BSpline<Scalar, 2, 3, true> curve;
            curve.set_control_points(ctrl_pts);
            curve.set_knots(knots);

            SECTION("Derivative")
            {
                validate_derivatives(curve, 10);
                validate_2nd_derivatives(curve, 10);
            }

            SECTION("Approximate inverse evaluate")
            {
                validate_approximate_inverse_evaluation(curve, 10);
            }

            SECTION("Split and combine")
            {
                const auto r = curve.convert_to_Bezier();
                decltype(curve) curve2(std::get<0>(r), std::get<1>(r));
                assert_same(curve, curve2, 10);
            }

            SECTION("Degree elevation")
            {
                const auto curve2 = curve.elevate_degree();
                assert_same(curve, curve2, 10);
            }
        }

        SECTION("Closed curve")
        {
            Eigen::Matrix<Scalar, 13, 2> ctrl_pts;
            ctrl_pts << 1, 4, .5, 6, 5, 4, 3, 12, 11, 14, 8, 4, 12, 3, 11, 9, 15, 10, 17, 8, 1, 4,
                .5, 6, 5, 4;
            Eigen::Matrix<Scalar, 17, 1> knots;
            knots << 0.0 / 16, 1.0 / 16, 2.0 / 16, 3.0 / 16, 4.0 / 16, 5.0 / 16, 6.0 / 16, 7.0 / 16,
                8.0 / 16, 9.0 / 16, 10.0 / 16, 11.0 / 16, 12.0 / 16, 13.0 / 16, 14.0 / 16,
                15.0 / 16, 16.0 / 16;

            BSpline<Scalar, 2, 3, true> curve;
            curve.set_control_points(ctrl_pts);
            curve.set_knots(knots);

            SECTION("Derivative")
            {
                validate_derivatives(curve, 10);
                validate_2nd_derivatives(curve, 10);
            }

            SECTION("Knot insertion and removal")
            {
                auto curve2 = curve;
                curve2.insert_knot(0.5, 2);
                assert_same(curve, curve2, 10);

                REQUIRE(curve2.remove_knot(0.5, 2) == 2);
                assert_same(curve, curve2, 10);
            }

            SECTION("Approximate inverse evaluate")
            {
                validate_approximate_inverse_evaluation(curve, 10);
            }

            SECTION("Turning angle")
            {
#if NANOSPLINE_SYMPY
                const auto min_t = curve.get_domain_lower_bound();
                const auto max_t = curve.get_domain_upper_bound();
                auto total_turning_angle = curve.get_turning_angle(min_t, max_t);
                REQUIRE(std::abs(total_turning_angle) == Approx(2 * M_PI));
#endif
            }

            SECTION("Singularity")
            {
#if NANOSPLINE_SYMPY
                auto singular_pts = curve.compute_singularities(0, 1);
                REQUIRE(singular_pts.size() == 0);
#endif
            }

            SECTION("Split and combine")
            {
                const auto r = curve.convert_to_Bezier();
                decltype(curve) curve2(std::get<0>(r), std::get<1>(r));
                assert_same(curve, curve2, 10);
            }

            SECTION("Degree elevation")
            {
                const auto curve2 = curve.elevate_degree();
                assert_same(curve, curve2, 10);
            }
        }
    }

    SECTION("Inflection")
    {
#if NANOSPLINE_SYMPY
        SECTION("Compare with Bezier")
        {
            Eigen::Matrix<Scalar, 4, 2> ctrl_pts;
            ctrl_pts << 0.0, 0.0, 1.0, 1.0, 2.0, -1.0, 3.0, 0.0;
            Eigen::Matrix<Scalar, 8, 1> knots;
            knots << 0, 0, 0, 0, 2, 2, 2, 2;

            BSpline<Scalar, 2, 3> curve;
            curve.set_control_points(ctrl_pts);
            curve.set_knots(knots);

            auto k = curve.evaluate_curvature(1.0).norm();
            REQUIRE(k == Approx(0).margin(1e-6));

            SECTION("Simple")
            {
                auto inflections = curve.compute_inflections(0, 2);
                REQUIRE(inflections.size() == 1);
                REQUIRE(inflections[0] == Approx(1.0));
            }

            SECTION("Inflection at knot value")
            {
                // Add a knot at the inflection point should not change anything.
                curve.insert_knot(1.0);
                auto inflections = curve.compute_inflections(0, 2);
                REQUIRE(inflections.size() == 1);
                REQUIRE(inflections[0] == Approx(1.0));
            }

            SECTION("Narrower range")
            {
                auto inflections = curve.compute_inflections(0.9, 1.1);
                REQUIRE(inflections.size() == 1);
                REQUIRE(inflections[0] == Approx(1.0));
            }

            SECTION("Inflection at boundary of the range")
            {
                auto inflections = curve.compute_inflections(0.0, 1.0);
                REQUIRE(inflections.size() == 1);
                REQUIRE(inflections[0] == Approx(1.0));

                inflections = curve.compute_inflections(1.0, 2.0);
                REQUIRE(inflections.size() == 1);
                REQUIRE(inflections[0] == Approx(1.0));
            }

            SECTION("Singularity")
            {
                auto singular_pts = curve.compute_singularities(0, 2.0);
                REQUIRE(singular_pts.size() == 0);
            }

            SECTION("Degree elevation")
            {
                const auto curve2 = curve.elevate_degree();
                assert_same(curve, curve2, 10);
            }

            SECTION("Update") { offset_and_validate(curve); }
        }

        SECTION("Closed curve")
        {
            Eigen::Matrix<Scalar, 14, 2> ctrl_pts;
            ctrl_pts << 1, 4, .5, 6, 5, 4, 3, 12, 11, 14, 8, 4, 12, 3, 11, 9, 15, 10, 17, 8, 1, 4,
                .5, 6, 5, 4, 3, 12;
            Eigen::Matrix<Scalar, 18, 1> knots;
            knots << 0.0 / 17, 1.0 / 17, 2.0 / 17, 3.0 / 17, 4.0 / 17, 5.0 / 17, 6.0 / 17, 7.0 / 17,
                8.0 / 17, 9.0 / 17, 10.0 / 17, 11.0 / 17, 12.0 / 17, 13.0 / 17, 14.0 / 17,
                15.0 / 17, 16.0 / 17, 17.0 / 17;

            BSpline<Scalar, 2, 3, true> curve;
            curve.set_control_points(ctrl_pts);
            curve.set_knots(knots);

            auto inflections = curve.compute_inflections(knots.minCoeff(), knots.maxCoeff());
            for (auto t : inflections) {
                auto k = curve.evaluate_curvature(t).norm();
                REQUIRE(k == Approx(0).margin(1e-6));
            }

            auto singular_pts = curve.compute_singularities(0, 2.0);
            REQUIRE(singular_pts.size() == 0);

            SECTION("Update") { offset_and_validate(curve); }
        }
#endif
    }

    SECTION("Extrapolation")
    {
        Eigen::Matrix<Scalar, 5, 2> control_pts;
        control_pts << 0.0, 0.0, 1.0, 1.0, 2.0, 0.5, 1.0, 0.0, 0.0, 1.0;
        BSpline<Scalar, 2, -1> curve;
        curve.set_control_points(control_pts);

        Eigen::Matrix<Scalar, 9, 1> knots;
        knots << 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 2.0;
        curve.set_knots(knots);

        auto p0 = curve.evaluate(curve.get_domain_lower_bound() - 0.1);
        auto p1 = curve.evaluate(curve.get_domain_upper_bound() + 0.1);
        REQUIRE(p0[0] < 0);
        REQUIRE(p0[1] < 0);
        REQUIRE(p1[0] < 0);
        REQUIRE(p1[1] > 1);

        auto d0 = curve.evaluate_derivative(curve.get_domain_lower_bound() - 0.1);
        auto d1 = curve.evaluate_derivative(curve.get_domain_upper_bound() + 0.1);
        REQUIRE(d0[0] > 0);
        REQUIRE(d0[1] > 0);
        REQUIRE(d1[0] < 0);
        REQUIRE(d1[1] > 0);

        auto dd0 = curve.evaluate_2nd_derivative(curve.get_domain_lower_bound() - 0.1);
        auto dd1 = curve.evaluate_2nd_derivative(curve.get_domain_upper_bound() + 0.1);
        REQUIRE(dd0[0] == Approx(dd1[0]));
        REQUIRE(dd0[1] == Approx(-dd1[1]));
    }

    SECTION("Periodic curve")
    {
        Eigen::Matrix<Scalar, 13, 2> ctrl_pts;
        ctrl_pts << 1, 4, .5, 6, 5, 4, 3, 12, 11, 14, 8, 4, 12, 3, 11, 9, 15, 10, 17, 8, 1, 4, .5,
            6, 5, 4;
        Eigen::Matrix<Scalar, 17, 1> knots;
        knots << 0.0 / 16, 1.0 / 16, 2.0 / 16, 3.0 / 16, 4.0 / 16, 5.0 / 16, 6.0 / 16, 7.0 / 16,
            8.0 / 16, 9.0 / 16, 10.0 / 16, 11.0 / 16, 12.0 / 16, 13.0 / 16, 14.0 / 16, 15.0 / 16,
            16.0 / 16;

        BSpline<Scalar, 2, 3, true> curve;
        curve.set_control_points(ctrl_pts);
        curve.set_knots(knots);

        REQUIRE(curve.is_closed());
        REQUIRE(curve.is_closed(1));
        REQUIRE(curve.is_closed(2));

        curve.set_periodic(true);

        const Scalar t_min = curve.get_domain_lower_bound();
        const Scalar t_max = curve.get_domain_upper_bound();
        const Scalar period = t_max - t_min;
        Eigen::Matrix<Scalar, 1, 2> p = curve.evaluate(t_min);
        Eigen::Matrix<Scalar, 1, 2> n = curve.evaluate_2nd_derivative(t_min).normalized();
        const Scalar s = 0.1;
        Eigen::Matrix<Scalar, 1, 2> q = p + s * n;

        Scalar t0 = curve.approximate_inverse_evaluate(q, t_min, t_max, 10);
        REQUIRE((curve.evaluate(t0) - p).norm() == Approx(s).margin(2e-2));
        Scalar t1 = curve.approximate_inverse_evaluate(q, t_min + period, t_max + period, 10);
        REQUIRE((curve.evaluate(t1) - p).norm() == Approx(s).margin(2e-2));
        Scalar t2 =
            curve.approximate_inverse_evaluate(q, t_min + period / 2, t_max + period / 2, 10);
        REQUIRE((curve.evaluate(t2) - p).norm() == Approx(s).margin(2e-2));
        Scalar t3 =
            curve.approximate_inverse_evaluate(q, t_min - period / 5, t_min + period / 5, 10);
        REQUIRE((curve.evaluate(t3) - p).norm() == Approx(s).margin(2e-2));

        auto curve2 = curve.elevate_degree();
        REQUIRE(curve2.get_periodic() == curve.get_periodic());
        assert_same(curve, curve2, 10);
    }

    SECTION("Debug")
    {
        BSpline<Scalar, 3, -1> curve;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 3> ctrl_pts(27, 3);
        ctrl_pts << -10.0, 15.506357108637001, 18.006511983541003, -10.0, 15.506357709411,
            18.006629839132, -10.0, 15.506358341393, 18.006747692976, -10.0, 15.50635900458,
            18.006865545062, -10.0, 15.506359698973998, 18.006983395381997, -10.0, 15.506360424569,
            18.007101243925998, -10.0, 15.506361181371, 18.007219090683, -10.0, 15.506361969366,
            18.007336935646002, -10.0, 15.506362788568, 18.007454778803, -10.0, 15.506363638962,
            18.007572620146, -10.0, 15.506364520557, 18.007690459664, -10.0, 15.506365433344001,
            18.007808297349, -10.0, 15.506366377325, 18.007926133191, -10.0, 15.506367352498,
            18.008043967179002, -10.0, 15.508255161015, 18.236155250047002, -10.0,
            15.627322787715999, 18.467795837419, -10.0, 15.868302636761, 18.632072043908, -10.0,
            16.197981254385, 18.659459626453, -10.0, 16.483391626235, 18.486770498043,
            -9.999999999989999, 16.707019854408, 18.241772633259, -10.0, 16.724266539943002,
            17.786982972154, -10.0, 16.497205399631998, 17.517344222271, -10.0, 16.200793677796,
            17.340680850414998, -10.0, 15.862443361514998, 17.372627297494, -10.0, 15.621200694973,
            17.543590743503, -10.0, 15.505194088991999, 17.778358880633, -10.0, 15.506357108637001,
            18.006511983541003;

        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> knots(41, 1);
        knots << -0.9994837030878214, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.000516296912178527, 0.000516296912178527, 0.000516296912178527,
            0.000516296912178527, 0.000516296912178527, 0.000516296912178527, 0.000516296912178527,
            0.000516296912178527, 0.000516296912178527, 0.000516296912178527, 0.000516296912178527,
            0.000516296912178527, 0.000516296912178527, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0005162969121786;
        curve.set_control_points(ctrl_pts);
        curve.set_knots(knots);
        curve.set_periodic(true);
        curve.initialize();

        REQUIRE(curve.is_closed());
        validate_derivatives(curve, 10);
        validate_2nd_derivatives(curve, 10);
    }
}
