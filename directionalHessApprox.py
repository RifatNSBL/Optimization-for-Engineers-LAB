# Optimization for Engineers - Dr.Johannes Hild
# Directional Hessian Approximation

# Purpose: Approximates Hessian times direction with central differences

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# x: column vector in R ** n(domain point)
# d: column vector in R ** n(search direction)
# delta: tolerance for termination. Default value: 1.0e-6
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# dH: Hessian times direction, column vector in R ** n

# Required files:
# < none >

# Test cases:
# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# x = np.array([[-1.01], [1]])
# d = np.array([[1], [1]])

# dH = directionalHessApprox(myObjective, x, d)
# should return dH = [[1.55491],[0]]

import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 23184261
    return matrnr


def directionalHessApprox(f, x: np.array, d: np.array, delta=1.0e-6, verbose=0):

    if verbose:
        print('Start directionalHessApprox...')

    d_norm = np.linalg.norm(d, 2)

    d_f_r = f.gradient(x + delta * d / d_norm)
    d_f_l = f.gradient(x - delta * d / d_norm)

    dH = d_norm / (2 * delta) * (d_f_r - d_f_l)

    if verbose:
        print('directionalHessApprox terminated with dH=', dH)

    return dH
