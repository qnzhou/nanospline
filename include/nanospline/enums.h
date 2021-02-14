#pragma once

namespace nanospline {

enum class CurveEnum {
    BEZIER = 1,
    RATIONAL_BEZIER = 2,
    BSPLINE = 3,
    NURBS = 4,
    LINE = 5,
    CIRCLE = 6,
    ELLIPSE = 7
};

enum class PatchEnum {
    BEZIER = 1,
    RATIONAL_BEZIER = 2,
    BSPLINE = 3,
    NURBS = 4
};

enum class SampleMethod {
    UNIFORM_DOMAIN = 1,
    UNIFORM_RANGE = 2,
    ADAPTIVE = 3
};

}
