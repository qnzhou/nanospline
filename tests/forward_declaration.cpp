#include <nanospline/Bezier.h>
#include <nanospline/RationalBezier.h>
#include <nanospline/BSpline.h>
#include <nanospline/NURBS.h>

namespace nanospline {

template class Bezier<double, 2, 0, false>;
template class Bezier<double, 2, 1, false>;
template class Bezier<double, 2, 2, false>;
template class Bezier<double, 2, 3, false>;
template class Bezier<double, 2, 4, false>;
template class Bezier<double, 2, 5, false>;

template class Bezier<double, 2, -1, true>;
template class Bezier<double, 2, 0, true>;
template class Bezier<double, 2, 1, true>;
template class Bezier<double, 2, 2, true>;
template class Bezier<double, 2, 3, true>;
template class Bezier<double, 2, 4, true>;
template class Bezier<double, 2, 5, true>;

template class RationalBezier<double, 2, 0, false>;
template class RationalBezier<double, 2, 1, false>;
template class RationalBezier<double, 2, 2, false>;
template class RationalBezier<double, 2, 3, false>;
template class RationalBezier<double, 2, 4, false>;
template class RationalBezier<double, 2, 5, false>;

template class RationalBezier<double, 2, -1, true>;
template class RationalBezier<double, 2, 0, true>;
template class RationalBezier<double, 2, 1, true>;
template class RationalBezier<double, 2, 2, true>;
template class RationalBezier<double, 2, 3, true>;
template class RationalBezier<double, 2, 4, true>;
template class RationalBezier<double, 2, 5, true>;

template class BSpline<double, 2, 0, false>;
template class BSpline<double, 2, 1, false>;
template class BSpline<double, 2, 2, false>;
template class BSpline<double, 2, 3, false>;
template class BSpline<double, 2, 4, false>;
template class BSpline<double, 2, 5, false>;

template class BSpline<double, 2, -1, true>;
template class BSpline<double, 2, 0, true>;
template class BSpline<double, 2, 1, true>;
template class BSpline<double, 2, 2, true>;
template class BSpline<double, 2, 3, true>;
template class BSpline<double, 2, 4, true>;
template class BSpline<double, 2, 5, true>;

template class NURBS<double, 2, 0, false>;
template class NURBS<double, 2, 1, false>;
template class NURBS<double, 2, 2, false>;
template class NURBS<double, 2, 3, false>;
template class NURBS<double, 2, 4, false>;
template class NURBS<double, 2, 5, false>;

template class NURBS<double, 2, -1, true>;
template class NURBS<double, 2, 0, true>;
template class NURBS<double, 2, 1, true>;
template class NURBS<double, 2, 2, true>;
template class NURBS<double, 2, 3, true>;
template class NURBS<double, 2, 4, true>;
template class NURBS<double, 2, 5, true>;

}
