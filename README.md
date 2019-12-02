# nanospline

Nanospline is a header-only spline library written with modern C++. It is
created by Qingnan Zhou as a coding exercise. It supports Beziér, rational
Beziér, B-spline and NURBS curves of _arbitrary_ degree in _arbitrary_
dimension. Most of the algorithms are covered by [The NURBS Book].

## Functionalities

The following functionalities are covered:

### Data structure

Nanospline provide 4 basic data structures for the 4 types of curves: 
Beziér, rational Beziér, B-spline and NURBS.  All of them are templated by 4
parameters:

* `Scalar`: The floating point data type.  (e.g. `float`, `double`, `long
        double`, etc.)
* `dim`: The dimension of the embedding space.  (e.g. `2` for 2D curves, and `3`
        for 3D curves.)
* `degree`: The degree of the curve.  (e.g. `2` for quardratic curves, `3` for
        cubic curves.)  The special value `-1` is used to indicate dynamic
        degree.
* `generic`: (Optional) This is a boolean flag indicating whether to treat the
        curve as a generic curve.  Nanospline sometimes provides specialized
        implementation when the degree of the curve is known.  By setting
        `generic` to `true`, we are forcing nanospline to use the default
        general implementation.  If `degree` is `-1` (i.e. dynamic degree),
        `generic` should always be true.

### Creation

#### Beziér curve

```c++
#include <nanospline/Bezier.h>

// Construct a 2D cubic Beziér curve
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

#### Rational Beziér curve

```c++
#include <nanospline/RationalBezier.h>

// Construct a 2D cubic rational Beziér curve
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

### Evaluation

### Derivatives

### Curvature

### Hodograph

### Inverse evaluation

### Knot insersion and splitting

[The NURBS Book]: https://www.springer.com/gp/book/9783642973857
