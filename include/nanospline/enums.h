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
    NURBS = 4,
    PLANE = 5,
    CYLINDER = 6,
    CONE = 7,
    SPHERE = 8,
    TORUS = 9,
    REVOLUTION = 10,
    EXTRUSION = 11
};

enum class SampleMethod {
    UNIFORM_DOMAIN = 1,
    UNIFORM_RANGE = 2,
    ADAPTIVE = 3
};

}
