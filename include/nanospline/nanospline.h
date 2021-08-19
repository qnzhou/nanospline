#pragma once

// Curve types.
#include <nanospline/BSpline.h>
#include <nanospline/Bezier.h>
#include <nanospline/NURBS.h>
#include <nanospline/RationalBezier.h>

// Patch types.
#include <nanospline/BSplinePatch.h>
#include <nanospline/BezierPatch.h>
#include <nanospline/NURBSPatch.h>
#include <nanospline/RationalBezierPatch.h>

// Curve primitive types.
#include <nanospline/Circle.h>
#include <nanospline/Ellipse.h>
#include <nanospline/Line.h>

// Patch primitive types.
#include <nanospline/Cone.h>
#include <nanospline/Cylinder.h>
#include <nanospline/ExtrusionPatch.h>
#include <nanospline/OffsetPatch.h>
#include <nanospline/Plane.h>
#include <nanospline/RevolutionPatch.h>
#include <nanospline/Sphere.h>
#include <nanospline/Torus.h>

// IO
#include <nanospline/load_msh.h>
#include <nanospline/save_msh.h>
#include <nanospline/save_obj.h>
#include <nanospline/save_svg.h>

// Utilities
#include <nanospline/arc_length.h>
#include <nanospline/conversion.h>
#include <nanospline/hodograph.h>
#include <nanospline/intersect.h>
#include <nanospline/sample.h>
#include <nanospline/split.h>
