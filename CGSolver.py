# Optimization for Engineers - Dr.Johannes Hild
# Conjugate Gradient Solver

# Purpose: CGSolver finds y such that norm(A * y - b) <= delta

# Input Definition:
# A: real valued spd matrix nxn
# b: column vector in R ** n
# delta: positive value, tolerance for termination. Default value: 1.0e-6.
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# x: column vector in R ^ n (solution in domain space)

# Required files:
# <none>


# Test cases:
# A = np.array([[4, 1, 0], [1, 7, 0], [ 0, 0, 3]], dtype=float)
# b = np.array([[5], [8], [3]], dtype=float)
# delta = 1.0e-6
# x = CGSolver(A, b, delta, 1)
# should return x = [[1], [1], [1]]

# A = np.array([[484, 374, 286, 176, 88], [374, 458, 195, 84, 3], [286, 195, 462, -7, -6], [176, 84, -7, 453, -10], [88, 3, -6, -10, 443]], dtype=float)
# b = np.array([[1320], [773], [1192], [132], [1405]], dtype=float)
# delta = 1.0e-6
# x = CGSolver(A, b, delta, 1)
# should return approx x = [[1], [0], [2], [0], [3]]


import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 0
    return matrnr


def CGSolver(A: np.array, b: np.array, delta=1.0e-6, verbose=0):

    if verbose:
        print('Start CGSolver...')

    countIter = 0

    x = b
    r = A @ x - b
    d = -r.copy()

    while np.linalg.norm(r) > delta:
        # INCOMPLETE CODE
        d_hash = np.dot(A, d)
        ro = np.dot(d.T, d_hash)
        t = np.linalg.norm(r, 2) ** 2 / ro
        x += t * d
        r_old = r.copy()
        r = r_old + t * d_hash
        betta = np.linalg.norm(r, 2) ** 2 / np.linalg.norm(r_old, 2) ** 2
        d = -r + betta * d
        countIter += 1
    if verbose:
        print('CGSolver terminated after ', countIter, ' steps with norm of residual being ', np.linalg.norm(r))

    return x
