# Nanospline

Nanospline is a header-only spline library written with modern C++. It is
created by Qingnan Zhou as a coding exercise. It supports Bézier, rational
Bézier, B-spline and NURBS curves of _arbitrary_ degree in _arbitrary_
dimension. Most of the algorithms are covered by [The NURBS Book].

[![](https://github.com/qnzhou/nanospline/workflows/build%20and%20test/badge.svg)](https://github.com/qnzhou/nanospline/actions)

## Functionalities

The following functionalities are covered:

* [Data structure](#Data-structure)
* [Creation](#Creation)
* [Basic information](#Basic-information)
* [Evaluation](#Evaluation)
* [Derivatives](#Derivatives)
* [Curvature](#Curvature)
* [Hodograph](#Hodograph)
* [Inverse evaluation](#Inverse-evaluation)
* [Knot insertion and removal](#Knot-insertion-and-removal)
* [Split](#Split)
* [Degree elevation](#Degree-elevation)
* [Inflection](#Inflection)
* [Turning angle](#Turning-angle)
* [Singularity](#Singularity)
* [Conversion](#Conversion)

### Data structure

Nanospline provide 4 basic data structures for the 4 types of curves:
Bézier, rational Bézier, B-spline and NURBS.  All of them are templated by 4
parameters:

* `Scalar`: The floating point data type.  (e.g. `float`, `double`, `long
        double`, etc.)
* `dim`: The dimension of the embedding space.  (e.g. `2` for 2D curves, and `3`
        for 3D curves.)
* `degree`: The degree of the curve.  (e.g. `2` for quadratic curves, `3` for
        cubic curves.)  The special value `-1` is used to indicate dynamic
        degree.
* `generic`: (Optional) This is a boolean flag indicating whether to treat the
        curve as a generic curve.  Nanospline sometimes provides specialized
        implementation when the degree of the curve is known.  By setting
        `generic` to `true`, we are forcing nanospline to use the default
        general implementation.  If `degree` is `-1` (i.e. dynamic degree),
        `generic` should always be true.

### Creation

All 4 types of curves can be constructed in the following pattern:

```c++
CurveType<Scalar, dim, degree, generic> curve;
```

where `CurveType` is one of the following: `Bezier`, `BSpline`, `RationalBezier`
and `NURBS`.  Different curve type requires setting different fields:

| Fields | Bézier | B-spline | Rational Bézier | NURBS |
|--------|--------|----------|-----------------|-------|
| Control points | Yes | Yes | Yes | Yes |
| Knots | No | Yes | No | Yes |
| Weights | No | No | Yes | Yes |

All fields are represented using `Eigen::Matrix` types.

#### Bézier curve

```c++
#include <nanospline/Bezier.h>

// Construct a 2D cubic Bézier curve
nanospline::Bezier<double, 2, 3> curve;

// Setting control points. Assuming `ctrl_pts` is a 4x2 Eigen matrix.
curve.set_control_points(ctrl_pts);
```

#### B-spline curve
```c++
#include <nanospline/BSpline.h>

// Construct a 2D cubic B-spline curve
nanospline::BSpline<double, 2, 3> curve;

// Setting control points. Assuming `ctrl_pts` is a nx2 Eigen matrix.
curve.set_control_points(ctrl_pts);

// Setting knots.  Assuming `knots` is a mx1 Eigen matrix.
// Where m = n+p+1, and `p` is the degree of the curve.
curve.set_knots(knots);
```

#### Rational Bézier curve

```c++
#include <nanospline/RationalBezier.h>

// Construct a 2D cubic rational Bézier curve
nanospline::RationalBezier<double, 2, 3> curve;

// Setting control points. Assuming `ctrl_pts` is a 4x2 Eigen matrix.
curve.set_control_points(ctrl_pts);

// Setting weights. Assuming `weights` is a 4x1 Eigen matrix.
curve.set_weights(weights);

// **Important**: RationalBezier requires initialization.
curve.initialize();
```

#### NURBS curve

```c++
#include <nanospline/NURBS.h>

// Construct a 2D cubic NURBS curve
nanospline::NURBS<double, 2, 3> curve;

// Setting control points. Assuming `ctrl_pts` is a 4x2 Eigen matrix.
curve.set_control_points(ctrl_pts);

// Setting knots.  Assuming `knots` is a mx1 Eigen matrix.
// Where m = n+p+1, and `p` is the degree of the curve.
curve.set_knots(knots);

// Setting weights. Assuming `weights` is a 4x1 Eigen matrix.
curve.set_weights(weights);

// **Important**: NURBS requires initialization.
curve.initialize();
```

### Basic information

Once a curve is initialized, one can query for a number of basic information:

```c++
// Polynomial degree.
int degree = curve.get_degree();

// The minimum and maximum parameter value.
auto t_min = curve.get_domain_lower_bound();
auto t_max = curve.get_domain_upper_bound();

// Control points
const auto& ctrl_pts = curve.get_control_points();

// Knots (BSpline and NURBS only).
const auto& knots = curve.get_knots();

// Weights (RationalBeizer and NURBS only).
const auto& weights = curve.get_weights();
```

### Evaluation

One can retrieve the point, `p`, corresponding to a given parameter value, `t`,
by evaluating the curve:

```c++
auto p = curve.evaluate(t);
```

### Derivatives

One can compute the first and second derivative vectors, `d1` and `d2`
respectively, at a given parameter value, `t`, by evaluating the curve
derivatives as the following:

```c++
auto d1 = curve.evaluate_derivative(t);
auto d2 = curve.evaluate_2nd_derivative(t);
```

### Curvature

Nanospline also provides direct support for computing curvature vector, `k`,
at a given parameter value, `t`, using the following:

```c++
auto k = curve.evaluate_curvature(t);
```

### Hodograph

For Bézier and B-spline, it is well known that their derivative curves are also
Bézier or B-spline curves respectively. The first derivative curve is called
hodograph, which can be computed with the following:

```c++
#include <nanospline/hodograph.h>

auto hodograph = nanospline::compute_hodograph(curve);
```

Higher order derivative curves can be obtained by computing the hodograph of a
hodograph recursively.

### Inverse evaluation

Inverse evaluation aims to find a point, parameterized by `t`, on a given curve
that is closest to a query point, `q`.  In the most general case, inverse
evaluation can be reduced to finding roots of high degree polynomial.  However,
closed formula often exist for lower degree curves.  Nanospline
provide two different methods for inverse evaluation:

```c++
// Method 1
auto t = curve.inverse_evaluate(q);

// Method 2
auto t = curve.approximate_inverse_evaluate(q);

// Method 2 complete signature
auto t = curve.approximate_inverse_evaluate(q, t_min, t_max, level);
```

The first method, `inverse_evaluate()`, tries to find the exact closest point by
solving high degree polynomial or apply closed formula for low degree curves.
It is currently under development.

In contrast, the second method, `approximate_inverse_evaluate()`, uses brute
force bisection method to find an approximate closest point.  The parameter
`t_min` and `t_max` specify the search domain, and `level` is the recursion
level.  Higher recursion level provides more accurate result.

### Knot insertion and removal

For B-spline and NURBS curves, one can insert extra knot, `t`, with
multiplicity, `m`, using the following:

```c++
curve.insert_knot(t, m);
```

To remove a knot `m` times:

```c++
curve.remove_knot(t, m);
```

Just to be complete, the full signature of the `remove_knot` function is

```c++
int num_removed = curve.remove_knot(t, m, tolerance);
```

Where the return value `num_removed` indicates how many times the knot is
removed, and `tolerance` specifies the max allowed change (L2 distance) in the
curve that a valid removal can introduce.

### Split

To split a curve into two halves at parameter value `t`:

```c++
#include <nanospline/split.h>

auto halves = nanospline::split(curve, t);
```

One can also split a B-Spline curve into a sequence of Bézier curves:

```c++
#include <nanospline/BSpline.h>

auto r = bspline.convert_to_Bezier();
const auto& bezier_segments = std::get<0>(r);
const auto& parameter_bounds = std::get<1>(r);
```

Where the `i`th Bézier curve covers the knot span `[parameter_bounds[i],
parameter_bounds[i+1]]`.  It is also possible recombine these Bézier segments to
form a B-spline curve:

```c++
BSpline<Scalar, dim, degree, generic> curve(
    bezier_segments, parameter_bounds);

// or with uniform knot span

BSpline<Scalar, dim, degree, generic> curve(bezier_segments);
```

Similarly, NURBS curve can be split into a sequence of rational Bézier curves:

```c++
#include <nanospline/BSpline.h>

auto r = nurbs.convert_to_Bezier();
const auto& rational_bezier_segments = std::get<0>(r);
const auto& parameter_bounds = std::get<1>(r);

// Re-combine rational Bézier back into NURBS

NURBS<Scalar, dim, degree, generic> curve(
    rational_bezier_segments, parameter_bounds);
```

### Degree elevation

It is often useful to increase the degree of a curve:

```c++
auto curve2 = curve.elevate_degree();
assert(curve2.get_degree() == curve.get_degree()+1);
```

### Inflection

Nanospline also supports computing 2D curve inflection
points, i.e. points with zero curvature.

```c++
#include <nanospline/inflection.h>

auto inflections = nanospline::compute_inflections(curve);
```

Where `inflections` is a vector of parameter values corresponding to inflection
points.

### Turning angle

Turning angle is the total curvature of a given curve.  It represents how much a
curve is bending.  In nanospline, turning angle computation is supported for 2D
curves:

```c++
auto turning_angle = curve.get_turning_angle(t0, t1);
```

which returns the turning angle (in radians) for the curve segment between `t0`
and `t1`.  It is often important to determine the locations, `tcs`, where
splitting the curve at `tcs` will reduce the turning angle by half for each
curve piece:

```c++
auto turning_angle = curve.get_turning_angle(t0, t1);
std::vector<Scalar> tcs = curve.reduce_turning_angle(t0, t1);

if (!tcs.empty()) {
    auto turning_angle_0 = curve.get_turning_angle(t0, tcs.front());
    assert(std::abs(turning_angle * 0.5 - turning_angle_0) < EPS);
    ...
}
```

### Singularity

Singularity points of a curve are defined as the locations at where the curve has
0 first derivative.  Singularity locations between `t0` and `t1` for 2D curves
can be computed in the following way:

```c++
std::vector<Scalar> singularites = curve.compute_singularities(t0, t1);
```

### Conversion

It is easy to convert a Bézier curve into a BSpline curve:

```c++
#include <nanospline/conversion.h>

auto bspline = nanospline::convert_to_BSpline(bezier);
```

It is also possible to convert a BSpline curve into a sequence of Bézier curves:

```c++
#include <nanospline/conversion.h>

auto beziers = nanospline::convert_to_Bezier(bspline);
for (const auto& curve : bezier) {
    // `curve` is a Bezier<...> curve.
}
```

To convert a Bézier into rational Bézier curve:

```c++
#include <nanospline/conversion.h>

auto rationa_bezier = nanospline::convert_to_RationalBezier(bezier);
```

To convert a BSpline into NURBS curve:

```c++
#include <nanospline/conversion.h>

auto nurbs = nanospline::convert_to_nurbs(bspline);
```

To convert a NURBS curve into a number of rational Bézier curves:

```c++
#include <nanospline/conversion.h>

auto rationa_beizers = nanospline::convert_to_RationalBezier(nurbs);
```

It is sometimes possible to convert a rational Bézier curve into a plane Bézier
curve, and convert a NURBS curves into a BSpline curve.
Such conversion is allowed when the all weights are the same.  Otherwise, an
exception will be raised.

```c++
#include <nanospline/conversion.h>

try {
    auto bezier = nanospline::convert_to_Bezier(rational_bezier);
    auto bspline = nanospline::convert_to_BSpline(nurbs);
} catch (const std::exception& e) {
    ...
}
```

[The NURBS Book]: https://www.springer.com/gp/book/9783642973857
