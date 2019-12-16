#pragma once
#include <nanospline/Bezier.h>
#include <nanospline/RationalBezier.h>
#include <nanospline/BSpline.h>
#include <nanospline/NURBS.h>

namespace nanospline {

extern template class Bezier<double, 2, 0, false>;
extern template class Bezier<double, 2, 1, false>;
extern template class Bezier<double, 2, 2, false>;
extern template class Bezier<double, 2, 3, false>;
extern template class Bezier<double, 2, 4, false>;
extern template class Bezier<double, 2, 5, false>;

extern template class Bezier<double, 2, -1, true>;
extern template class Bezier<double, 2, 0, true>;
extern template class Bezier<double, 2, 1, true>;
extern template class Bezier<double, 2, 2, true>;
extern template class Bezier<double, 2, 3, true>;
extern template class Bezier<double, 2, 4, true>;
extern template class Bezier<double, 2, 5, true>;

extern template class RationalBezier<double, 2, 0, false>;
extern template class RationalBezier<double, 2, 1, false>;
extern template class RationalBezier<double, 2, 2, false>;
extern template class RationalBezier<double, 2, 3, false>;
extern template class RationalBezier<double, 2, 4, false>;
extern template class RationalBezier<double, 2, 5, false>;

extern template class RationalBezier<double, 2, -1, true>;
extern template class RationalBezier<double, 2, 0, true>;
extern template class RationalBezier<double, 2, 1, true>;
extern template class RationalBezier<double, 2, 2, true>;
extern template class RationalBezier<double, 2, 3, true>;
extern template class RationalBezier<double, 2, 4, true>;
extern template class RationalBezier<double, 2, 5, true>;

extern template class BSpline<double, 2, 0, false>;
extern template class BSpline<double, 2, 1, false>;
extern template class BSpline<double, 2, 2, false>;
extern template class BSpline<double, 2, 3, false>;
extern template class BSpline<double, 2, 4, false>;
extern template class BSpline<double, 2, 5, false>;

extern template class BSpline<double, 2, -1, true>;
extern template class BSpline<double, 2, 0, true>;
extern template class BSpline<double, 2, 1, true>;
extern template class BSpline<double, 2, 2, true>;
extern template class BSpline<double, 2, 3, true>;
extern template class BSpline<double, 2, 4, true>;
extern template class BSpline<double, 2, 5, true>;

extern template class NURBS<double, 2, 0, false>;
extern template class NURBS<double, 2, 1, false>;
extern template class NURBS<double, 2, 2, false>;
extern template class NURBS<double, 2, 3, false>;
extern template class NURBS<double, 2, 4, false>;
extern template class NURBS<double, 2, 5, false>;

extern template class NURBS<double, 2, -1, true>;
extern template class NURBS<double, 2, 0, true>;
extern template class NURBS<double, 2, 1, true>;
extern template class NURBS<double, 2, 2, true>;
extern template class NURBS<double, 2, 3, true>;
extern template class NURBS<double, 2, 4, true>;
extern template class NURBS<double, 2, 5, true>;

}
