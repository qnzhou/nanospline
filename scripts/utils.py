from sympy import symbols
from scipy.special import comb
import numpy as np

def indent(lines):
    return ["    {}".format(l) for l in lines];

def bezier(degree, t, rational, syms):
    tot = 0
    n = degree

    denom = 0
    for i in range(n+1):
        if rational:
            tot += int(comb(n, i)) * (1-t)**(n-i) * \
                t**i * syms[i][0] * syms[i][1]
        else:
            tot += int(comb(n, i)) * (1-t)**(n-i) * t**i * syms[i][0]
        denom += int(comb(n, i)) * (1-t)**(n-i) * t**i * syms[i][1]

    if rational:
        return tot/denom

    return tot


def setup_functions(maxb=11, maxrb=5):
    functions = {}

    beziers = []

    for i in range(2, maxb):
        beziers.append((i+1, i, False, lambda t, syms,
                        i=i: bezier(i, t, False, syms)))

    functions["Bezier"] = beziers

    rbeziers = []

    for i in range(2, maxrb):
        rbeziers.append((i+1, i, True, lambda t, syms,
                         i=i: bezier(i, t, True, syms)))

    functions["RationalBezier"] = rbeziers

    return functions


def create_coeff_symbols(n_coeffs):
    syms = []

    for i in range(n_coeffs):
        syms.append((
            np.array([
                symbols('cx{}'.format(i)),
                symbols('cy{}'.format(i))]),
            symbols('w{}'.format(i))))

    return syms


def generate_solver_code(n_coeffs, coeffs, is_rational, poly, printer):
    lines = []

    for i in range(n_coeffs):
        lines.append("Scalar cx{0} = ctrl_pts({0}, 0);".format(i))
        lines.append("Scalar cy{0} = ctrl_pts({0}, 1);".format(i))
        if is_rational:
            lines.append("Scalar w{0} = weights({0});".format(i))

    lines.append(
        "PolynomialRootFinder<Scalar, {}>::find_real_roots_in_interval({{".format(poly.degree()))

    lines.append(",\n".join(
        ["    {}".format(printer.doprint(c)) for c in coeffs]))
    lines.append("},")
    lines.append("result, t0, t1, tol);")

    return lines
