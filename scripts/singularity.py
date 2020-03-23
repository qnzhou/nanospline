#!/usr/bin/env python
# coding=utf-8
import utils
import os
from sympy import symbols, diff, together, fraction, Poly, Abs
from scipy.special import comb
import numpy as np
from sympy.printing.ccode import C99CodePrinter

code_template_header = """/**
 * This code is automatically generated by scripts/singularity.py
 */

#pragma once
#include <cassert>
#include <vector>

#include <nanospline/Exceptions.h>
#include <nanospline/PolynomialRootFinder.h>
#include <nanospline/BezierBase.h>

namespace nanospline {
namespace internal {

"""

code_template = """
template<typename Derived>
std::vector<typename Derived::Scalar> compute_singularities(
        const Eigen::PlainObjectBase<Derived>& ctrl_pts,
        typename Derived::Scalar t0 = 0,
        typename Derived::Scalar t1 = 1) {{
    switch(ctrl_pts.rows()-1) {{
{body}
    }}
}}
"""

specialization_template = """
template<typename Scalar>
std::vector<Scalar> compute_degree_{degree}_singularities(
        {control_variables},
        Scalar t0 = 0,
        Scalar t1 = 1) {{
    std::vector<Scalar> result;
    constexpr Scalar tol = static_cast<Scalar>(1e-8);

{body}

    return result;
}}

"""

specialization_extern_declaration_template = """
#if defined(HIGH_DEGREE_SUPPORT) || {degree} < 5
#define Scalar double
extern template
std::vector<Scalar> compute_degree_{degree}_singularities(
        {control_variables},
        Scalar t0,
        Scalar t1);
#undef Scalar

#if {degree} < 10
#define Scalar float
extern template
std::vector<Scalar> compute_degree_{degree}_singularities(
        {control_variables},
        Scalar t0,
        Scalar t1);
#undef Scalar
#endif
#endif
"""

specialization_extern_definition_template = """
#if defined(HIGH_DEGREE_SUPPORT) || {degree} < 5
#define Scalar double
template
std::vector<Scalar> compute_degree_{degree}_singularities(
        {control_variables},
        Scalar t0,
        Scalar t1);
#undef Scalar

#if {degree} < 10
#define Scalar float
template
std::vector<Scalar> compute_degree_{degree}_singularities(
        {control_variables},
        Scalar t0,
        Scalar t1);
#undef Scalar
#endif
#endif
"""

code_template_footer = """
} // End internal namespace
} // End nanospline namespace

"""

extern_declaration_template = """/**
 * This code is automatically generated by scripts/singularity.py
 */

#pragma once
#include <cassert>
#include <vector>

#include <nanospline/internal/auto_singularity.h>

namespace nanospline {{
namespace internal {{

{body}

}} // End internal namespace
}} // End nanospline namespace

"""

extern_definition_template = """/**
 * This code is automatically generated by scripts/singularity.py
 */

#include <cassert>
#include <vector>

#include "forward_declaration.h"

namespace nanospline {{
namespace internal {{

{body}

}} // End internal namespace
}} // End nanospline namespace

"""

def extract_coefficients(func, t):
    tx = (diff(func[0], t))
    ty = (diff(func[1], t))

    eq = tx*tx + ty*ty
    poly = Poly(eq, t)
    coeffs = poly.all_coeffs()
    coeffs.reverse()

    return poly, coeffs

def generate_code_for_Bezier(degree, printer):
    t = symbols('t')
    lines = []
    n_coeffs = degree+1
    syms = utils.create_coeff_symbols(n_coeffs)
    func = utils.bezier(degree, t, False, syms)

    poly, coeffs = extract_coefficients(func, t);
    solver_lines = utils.generate_solver_code(coeffs, poly, printer)

    control_variables = ", ".join(["Scalar cx{i}, Scalar cy{i}".format(i=i)
        for i in range(n_coeffs)])

    control_variable_arguments = ", ".join([
        "ctrl_pts({i},0), ctrl_pts({i},1)".format(i=i)
        for i in range(n_coeffs) ])

    template_function = specialization_template.format(
            control_variables = control_variables,
            body="\n".join(utils.indent(solver_lines)),
            degree=degree)

    invocation = "return compute_degree_{}_singularities({}, t0, t1);".format(
                        degree, control_variable_arguments)

    extern_declaration = specialization_extern_declaration_template.format(
            degree=degree,
            control_variables = control_variables)

    extern_definition = specialization_extern_definition_template.format(
            degree=degree,
            control_variables = control_variables)

    return template_function, invocation, extern_declaration, extern_definition;

if __name__ == "__main__":
    functions = utils.setup_functions()
    t = symbols('t')

    printer = C99CodePrinter()

    extern_declarations = [];
    extern_definitions = [];

    for poly_name in functions:
        code = code_template_header
        specialized_code = ""
        lines = []
        for n_coeffs, degree, is_rational, curve in functions[poly_name]:
            if is_rational:
                continue;
            else:
                template_function, invocation, extern_declaration, extern_definition = \
                generate_code_for_Bezier(degree, printer);


            if degree >= 5:
                lines.append("#ifdef HIGH_DEGREE_SUPPORT")
            lines.append("case {}:".format(degree))
            lines.append("    " + invocation);
            if degree >= 5:
                lines.append("#endif // HIGH_DEGREE_SUPPORT")

            specialized_code += template_function;
            extern_declarations.append(extern_declaration);
            extern_definitions.append(extern_definition);

        lines.append("default:")
        lines.append("    throw not_implemented_error(")
        lines.append("        \"Singularity computation only works on {} curve with degree lower than {}\");".format(poly_name, degree))

        body = "\n".join(utils.indent(utils.indent(lines)))
        code += specialized_code
        code += code_template.format(type=poly_name, body=body) + "\n\n"

        code += code_template_footer
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, "..", "include", "nanospline",
            "internal", "auto_singularity.h"), "w") as f:
            f.write(code)

    body = extern_declaration_template.format(body=("\n".join(extern_declarations)));
    with open(os.path.join(dir_path, "..", "tests",
        "forward_declaration_singularity.h"), "w") as f:
        f.write(body)

    body = extern_definition_template.format(body=("\n".join(extern_definitions)));
    with open(os.path.join(dir_path, "..", "tests",
        "forward_declaration_singularity.cpp"), "w") as f:
        f.write(body)
