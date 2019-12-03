import os
from sympy import *
from scipy.special import comb
import numpy as np
from sympy.printing.ccode import C99CodePrinter




def bezier(degree, t, sysms):
    tot = 0
    n = degree

    for i in range(n+1):
        tot += comb(n, i) * (1-t)**(n-i) * t**i * syms[i]

    return tot


def setup_functions():
    functions = {}

    beziers = []

    for i in range(2, 11):
        beziers.append((i+1, i, lambda t, syms, i=i: bezier(i, t, syms)))

    functions["Bezier"] = beziers

    return functions


def create_coeff_symbols(n_coeffs):
    syms = []

    for i in range(n_coeffs):
        syms.append(np.array([
            symbols('cx{}'.format(i)),
            symbols('cy{}'.format(i))]))

    return syms


if __name__ == "__main__":
    functions = setup_functions()
    t = symbols('t')

    code = ""
    code += "#pragma once\n\n"

    code +="#include <cassert>\n"
    code +="#include <vector>\n\n"

    code +="#include <nanospline/Exceptions.h>\n"
    code +="#include <nanospline/PolynomialRootFinder.h>\n"
    code +="#include <nanospline/Bezier.h>\n\n\n"

    code += "namespace nanospline {\n"

    printer = C99CodePrinter()

    for poly in functions:
        print("-----------------\n",poly)
        code += "template<typename Scalar, int _degree=3, bool generic=_degree<0 >\n"
        code += "std::vector<Scalar> compute_inflections(const {}<Scalar, 2, _degree, generic>& curve, Scalar t0 = 0, Scalar t1 = 1) {{\n".format(poly)
        code += "\tstd::vector<Scalar> result;\n\tconstexpr Scalar tol = 1e-8;\n"
        first=True
        for tmp in functions[poly]:
            n_coeffs = tmp[0]
            degree = tmp[1]
            syms = create_coeff_symbols(n_coeffs)
            func = tmp[2](t, syms)
            print(degree)


            tx = (diff(func[0], t))
            ty = (diff(func[1], t))

            nx = -(diff(func[1], t, 2))
            ny = (diff(func[0], t, 2))

            eq = (tx*nx+ty*ny)

            poly = Poly(eq, t)
            coeffs = poly.all_coeffs()
            coeffs.reverse()

            code += "\t{}if(curve.get_degree() == {}){{\n".format("" if first else "else ", degree)
            code += "\t\tconst auto& ctrl_pts = curve.get_control_points();\n"

            first = False

            for i in range(n_coeffs):
                code += "\t\tScalar cx{0} = ctrl_pts({0}, 0);\n".format(i)
                code += "\t\tScalar cy{0} = ctrl_pts({0}, 1);\n".format(i)

            code += "\t\tPolynomialRootFinder<Scalar, {}>::find_real_roots_in_interval({{\n".format(poly.degree())

            for c in coeffs:
                code += "\t\t\t{},\n".format(printer.doprint((c)))

            code = code[0:-2]
            code += "},\n\t\tresult, t0, t1, tol);\n"

            code += "\t}\n"

        code += '\telse{\n\t\tthrow not_implemented_error("Inflection computation only works on cubic Bézier curve with degree lower than 10");\n\t}\n'
        code += "\treturn result;\n"
        code += "}\n\n"

    code += "}\n"

    dir_path = os.path.dirname(os.path.realpath(__file__))

    with open(os.path.join(dir_path, "..", "include", "nanospline", "inflection.h"), "w") as f:
        f.write(code)
