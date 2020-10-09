#include <catch2/catch.hpp>
#include <iostream>
#include <ostream>
#include <vector>

#include <nanospline/BSpline.h>
#include <nanospline/BSplinePatch.h>
#include <nanospline/Bezier.h>
#include <nanospline/BezierPatch.h>
#include <nanospline/forward_declaration.h>
#include <nanospline/save_svg.h>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "validation_utils.h"

using namespace nanospline;
using Scalar = double;
Eigen::VectorXd get_closed_equispaced_knots(int num_knots, int degree)
{
    Eigen::VectorXd knots(num_knots, 1);
    knots.segment(0, degree).setConstant(0.);
    knots.segment(num_knots - degree, degree).setConstant(1.);

    int num_internal_knots = num_knots - 2 * degree;
    Eigen::MatrixXd internal_knots = Eigen::ArrayXd::LinSpaced(num_internal_knots, 0., 1.);
    knots.segment(degree, num_internal_knots) = internal_knots;
    return knots;
}

template <int dim, int degree, bool generic>
void test_curve_fit(
    Bezier<Scalar, dim, degree, generic> curve, int num_tests = 5, int num_control_pts = degree + 1)
{
    for (int i = 0; i < num_tests; i++) {
        const int num_samples = degree * degree;

        Bezier<Scalar, dim, degree, false> true_curve;

        Eigen::Matrix<Scalar, degree + 1, dim> control_pts;
        control_pts = Eigen::MatrixXd::Random(degree + 1, dim);
        true_curve.set_control_points(control_pts);


        Eigen::Matrix<Scalar, num_samples, dim> curve_samples;
        Eigen::Matrix<Scalar, num_samples, 1> parameter_values;
        parameter_values = .5 * (Eigen::MatrixXd::Random(num_samples, 1) +
                                    Eigen::MatrixXd::Constant(num_samples, 1, 1.));

        for (int i = 0; i < num_samples; i++) {
            Scalar t = parameter_values(i, 0);
            curve_samples.row(i) = true_curve.evaluate(t);
        }

        auto computed_curve =
            Bezier<Scalar, dim, degree, false>::fit(parameter_values, curve_samples);
        assert_same(computed_curve, true_curve, num_samples, 0., 1., 0., 1., 1e-5);
    }
}

template <int dim, int degree, bool generic>
void test_curve_fit(
    BSpline<Scalar, dim, degree, generic> curve, int num_control_pts, int num_tests = 5)
{
    // const int num_control_pts = degree + 1;
    int num_knots = num_control_pts + degree + 1;

    Eigen::VectorXd knots= get_closed_equispaced_knots(num_knots, degree);
    for (int i = 0; i < num_tests; i++) {
        const int num_samples = num_control_pts * num_control_pts;

        BSpline<Scalar, dim, degree, false> true_curve;

        Eigen::MatrixXd control_pts(num_control_pts, dim);
        control_pts = Eigen::MatrixXd::Random(num_control_pts, dim);

        true_curve.set_control_points(control_pts);
        true_curve.set_knots(knots);

        Eigen::MatrixXd curve_samples(num_samples, dim);
        Eigen::MatrixXd parameter_values(num_samples, 1);
        parameter_values = .5 * (Eigen::MatrixXd::Random(num_samples, 1) +
                                    Eigen::MatrixXd::Constant(num_samples, 1, 1.));

        for (int i = 0; i < num_samples; i++) {
            Scalar t = parameter_values(i, 0);
            curve_samples.row(i) = true_curve.evaluate(t);
        }

        auto computed_curve = BSpline<Scalar, dim, degree, false>::fit(
            parameter_values, curve_samples, num_control_pts, knots);
        assert_same(computed_curve, true_curve, num_samples, 0., 1., 0., 1., 1e-5);
    }
}
template <int dim, int degree_u, int degree_v>
void test_patch_fit(BezierPatch<Scalar, dim, degree_u, degree_v> curve, int num_tests = 3)
{
    for (int i = 0; i < num_tests; i++) {
        const int max_degree = degree_u * degree_v;
        const int num_samples = 2 * max_degree;
        const int num_control_pts = (degree_u + 1) * (degree_v + 1);

        BezierPatch<Scalar, dim, degree_u, degree_v> true_patch;

        Eigen::Matrix<Scalar, num_control_pts, dim> control_pts;
        control_pts = Eigen::MatrixXd::Random(num_control_pts, dim);
        true_patch.set_control_grid(control_pts);
        true_patch.initialize();


        Eigen::Matrix<Scalar, num_samples, dim> patch_samples;
        Eigen::Matrix<Scalar, num_samples, 2> parameter_values;
        parameter_values = .5 * (Eigen::MatrixXd::Random(num_samples, 2) +
                                    Eigen::MatrixXd::Constant(num_samples, 2, 1.));

        for (int i = 0; i < num_samples; i++) {
            Scalar u = parameter_values(i, 0);
            Scalar v = parameter_values(i, 1);
            patch_samples.row(i) = true_patch.evaluate(u, v);
        }

        auto computed_patch =
            BezierPatch<Scalar, dim, degree_u, degree_v>::fit(parameter_values, patch_samples);
        assert_same(computed_patch, true_patch, num_samples, 0., 1., 0., 1., 0., 1., 0., 1., 1e-5);
    }
}

template <int dim, int degree_u, int degree_v>
void test_patch_fit(BSplinePatch<Scalar, dim, degree_u, degree_v> curve,
    int num_control_pts_u,
    int num_control_pts_v,
    int num_tests = 3)
{
    // const int num_control_pts = degree + 1;
    int num_knots_u = num_control_pts_u + degree_u + 1;
    int num_knots_v = num_control_pts_v + degree_v + 1;

    Eigen::VectorXd knots_u = get_closed_equispaced_knots(num_knots_u, degree_u);
    Eigen::VectorXd knots_v = get_closed_equispaced_knots(num_knots_v, degree_v);
    for (int i = 0; i < num_tests; i++) {
        int num_control_pts = num_control_pts_u * num_control_pts_v;
        const int num_samples = 2 * num_control_pts;

        BSplinePatch<Scalar, dim, degree_u, degree_v> true_patch;

        Eigen::MatrixXd control_pts(num_control_pts, dim);
        control_pts = Eigen::MatrixXd::Random(num_control_pts, dim);

        true_patch.set_control_grid(control_pts);
        true_patch.set_knots_u(knots_u);
        true_patch.set_knots_v(knots_v);
        true_patch.initialize();

        Eigen::MatrixXd patch_samples(num_samples, dim);
        Eigen::MatrixXd parameter_values(num_samples, 2);
        parameter_values = .5 * (Eigen::MatrixXd::Random(num_samples, 2) +
                                    Eigen::MatrixXd::Constant(num_samples, 2, 1.));

        for (int i = 0; i < num_samples; i++) {
            Scalar u = parameter_values(i, 0);
            Scalar v = parameter_values(i, 1);
            patch_samples.row(i) = true_patch.evaluate(u, v);
        }

        auto computed_patch = 
            BSplinePatch<Scalar, dim, degree_u, degree_v>::fit(parameter_values,
            patch_samples, num_control_pts_u, num_control_pts_v,
            knots_u, knots_v);
        assert_same(computed_patch, true_patch, num_samples, 0., 1., 0., 1., 0., 1., 0., 1., 1e-5);
    }
}


TEST_CASE("test least squares fitting for Bezier curves", "[nonrational][bezier][fit]")
{
    SECTION("random polynomial fitting: dim 2, degree 4")
    {
        Bezier<Scalar, 2, 4, false> curve;
        test_curve_fit(curve);
    }
    SECTION("random polynomial fitting: dim 3, degree 15")
    {
        Bezier<Scalar, 3, 15, false> curve;
        test_curve_fit(curve);
    }
    SECTION("random polynomial fitting: dim 10, degree 10")
    {
        Bezier<Scalar, 10, 4, false> curve;
        test_curve_fit(curve);
    }
}

TEST_CASE("test least squares fitting for BSpline curves", "[nonrational][bspline][fit]")
{
    SECTION("random polynomial fitting: dim 2, degree 4")
    {
        BSpline<Scalar, 2, 4, false> curve;
        test_curve_fit(curve, 5);
        test_curve_fit(curve, 8);
        test_curve_fit(curve, 11);
    }
    SECTION("random polynomial fitting: dim 3, degree 15")
    {
        BSpline<Scalar, 3, 15, false> curve;
        test_curve_fit(curve, 18);
    }
    SECTION("random polynomial fitting: dim 10, degree 5")
    {
        BSpline<Scalar, 10, 5, false> curve;
        test_curve_fit(curve, 6);
    }
}


TEST_CASE("test least squares fitting for Bezier patches", "[nonrational][bezier_patch][fit]")
{
    SECTION("random polynomial fitting: dim 2, degree 4")
    {
        BezierPatch<Scalar, 2, 4, 4> patch;
        test_patch_fit(patch);
    }
    SECTION("random polynomial fitting: dim 3, mixed degree")
    {
        BezierPatch<Scalar, 3, 9, 7> patch;
        test_patch_fit(patch);
    }
    SECTION("random polynomial fitting: dim 5, mixed degree")
    {
        BezierPatch<Scalar, 5, 4, 10> patch;
        test_patch_fit(patch);
    }
}


TEST_CASE("test least squares fitting for BSpline patches", "[nonrational][bspline_patch][fit]")
{
    SECTION("random polynomial fitting: dim 2, degree 4")
    {
        BSplinePatch<Scalar, 2, 4, 4> patch;
        test_patch_fit(patch, 6, 7);
        test_patch_fit(patch, 8, 5);
    }
    SECTION("random polynomial fitting: dim 3, mixed degree")
    {
        BSplinePatch<Scalar, 3, 9, 7> patch;
        test_patch_fit(patch, 11, 9);
    }
}

TEST_CASE("simple test for deformations to test for seg faults", "[deform]")
{
    SECTION("Curves")
    {
        int num_control_pts;
        const int degree = 4;
        const int dim = 2;
        const int num_samples = 30;
        double magnitude = 10;

        Eigen::MatrixXd parameter_values = .5 * (Eigen::MatrixXd::Random(num_samples, 1) +
                                                    Eigen::MatrixXd::Constant(num_samples, 1, 1.));
        Eigen::MatrixXd deformations = Eigen::MatrixXd::Constant(num_samples, dim, magnitude);

        SECTION("Bezier")
        {
            Bezier<Scalar, dim, degree, false> curve;
            num_control_pts = degree + 1;

            Eigen::MatrixXd control_pts = Eigen::MatrixXd::Random(num_control_pts, dim);
            curve.set_control_points(control_pts);
            curve.deform(parameter_values, deformations);
            auto new_control_points = curve.get_control_points();
            REQUIRE((new_control_points - control_pts).maxCoeff() == Approx(magnitude));
            REQUIRE((new_control_points - control_pts).minCoeff() == Approx(magnitude));
        }
        SECTION("BSpline")
        {
            BSpline<Scalar, 2, degree, false> curve;
            num_control_pts = 3 * degree;
            int num_knots = num_control_pts + degree + 1;

            Eigen::VectorXd knots = get_closed_equispaced_knots(num_knots, degree);
            Eigen::MatrixXd control_pts = Eigen::MatrixXd::Random(num_control_pts, dim);
            curve.set_control_points(control_pts);
            curve.set_knots(knots);
            curve.deform(parameter_values, deformations);
            auto new_control_points = curve.get_control_points();
            REQUIRE((new_control_points - control_pts).maxCoeff() == Approx(magnitude));
            REQUIRE((new_control_points - control_pts).minCoeff() == Approx(magnitude));
        }
    }
    SECTION("Patches")
    {
        int num_control_pts_u;
        int num_control_pts_v;
        const int degree_u = 4;
        const int degree_v = 5;
        const int dim = 2;
        const int num_samples = 200;

        Eigen::MatrixXd parameter_values = .5 * (Eigen::MatrixXd::Random(num_samples, 2) +
                                                    Eigen::MatrixXd::Constant(num_samples, 2, 1.));
        Eigen::MatrixXd deformations = Eigen::MatrixXd::Constant(num_samples, dim, 1.);

        SECTION("Bezier Patch")
        {
            BezierPatch<Scalar, dim, degree_u, degree_v> patch;
            num_control_pts_u = degree_u + 1;
            num_control_pts_v = degree_v + 1;
            Eigen::MatrixXd control_pts =
                Eigen::MatrixXd::Random(num_control_pts_u * num_control_pts_v, dim);
            patch.set_control_grid(control_pts);
            patch.initialize();
            patch.deform(parameter_values, deformations);
            auto new_control_points = patch.get_control_grid();
            int num_control_pts = int(new_control_points.rows());
            REQUIRE(new_control_points.isApprox(
                control_pts + Eigen::MatrixXd::Constant(num_control_pts, dim, 1)));
        }
        SECTION("BSpline Patch")
        {
            BSplinePatch<Scalar, dim, degree_u, degree_v> patch;
            num_control_pts_u = 2 * degree_u;
            num_control_pts_v = 2 * degree_v;
            int num_knots_u = num_control_pts_u + degree_u + 1;
            int num_knots_v = num_control_pts_v + degree_v + 1;
            Eigen::VectorXd knots_u = get_closed_equispaced_knots(num_knots_u, degree_u);
            Eigen::VectorXd knots_v = get_closed_equispaced_knots(num_knots_v, degree_v);
            Eigen::MatrixXd control_pts =
                Eigen::MatrixXd::Random(num_control_pts_u * num_control_pts_v, dim);
            patch.set_control_grid(control_pts);
            patch.set_knots_u(knots_u);
            patch.set_knots_v(knots_v);
            patch.initialize();
            patch.deform(parameter_values, deformations);
            auto new_control_points = patch.get_control_grid();
            int num_control_pts = int(new_control_points.rows());
            REQUIRE(new_control_points.isApprox(
                control_pts + Eigen::MatrixXd::Constant(num_control_pts, dim, 1)));
        }
    }
}
