# Optimization for Engineers - Dr.Johannes Hild
# Wolfe-Powell line search

# Purpose: Find t to satisfy f(x+t*d)<=f(x) + t*sigma*gradf(x).T@d and gradf(x+t*d).T@d >= rho*gradf(x).T@d

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# x: column vector in R ** n(domain point)
# d: column vector in R ** n(search direction)
# sigma: value in (0, 1 / 2), marks quality of decrease. Default value: 1.0e-3
# rho: value in (sigma, 1), marks quality of steepness. Default value: 1.0e-2
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# t: t is set, such that t satisfies both Wolfe - Powell conditions

# Required files:
# < none >

# Test cases:
# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# x = np.array([[-1.01], [1]])
# d = np.array([[1], [1]])
# sigma = 1.0e-3
# rho = 1.0e-2
# t = WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
# should return t=1

# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# x = np.array([[-1.2], [1]])
# d = np.array([[0.1], [1]])
# sigma = 1.0e-3
# rho = 1.0e-2
# t = WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
# should return t=16

# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# x = np.array([[-0.2], [1]])
# d = np.array([[1], [1]])
# sigma = 1.0e-3
# rho = 1.0e-2
# t = WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
# should return t=0.25

# myObjective = nonlinearObjective()
# x = np.array([[0.53], [-0.29]])
# d = np.array([[-3.88], [1.43]])
# sigma = 1.0e-3
# rho = 1.0e-2
# t = WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
# should return t=0.0938


import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 0
    return matrnr


def W1_check(f, x: np.array, d: np.array, descent, t, sigma):
    return f.objective(x + t * d) <= f.objective(x) + t * sigma * descent


def W2_check(f, x: np.array, d: np.array, descent, t, rho):
    return np.dot(f.gradient(x + t * d).T, d) >= rho * descent


def WolfePowellSearch(f, x: np.array, d: np.array, sigma=1.0e-3, rho=1.0e-2, verbose=0):
    fx = f.objective(x)
    gradx = f.gradient(x)
    descent = gradx.T @ d

    if descent >= 0:
        raise TypeError('descent direction check failed!')

    if sigma <= 0 or sigma >= 0.5:
        raise TypeError('range of sigma is wrong!')

    if rho <= sigma or rho >= 1:
        raise TypeError('range of rho is wrong!')

    if verbose:
        print('Start WolfePowellSearch...')

    t = 1

    # INCOMPLETE CODE

    if np.dot(gradx.T, d) >= 0: return -1

    W1 = W1_check(f, x, d, descent, t, sigma)
    W2 = W2_check(f, x, d, descent, t, rho)

    t_minus, t_plus = 0, 0
    if not W1:
        t /= 2
        while not W1_check(f, x, d, descent, t, sigma):
            t /= 2
        t_minus = t
        t_plus = 2 * t

    elif W2:
        return t

    else:  # Fronttracking
        t *= 2
        while W1_check(f, x, d, descent, t, sigma):
            t *= 2
        t_minus = t / 2
        t_plus = t

    t = t_minus
    W1 = W1_check(f, x, d, descent, t, sigma)
    W2 = W2_check(f, x, d, descent, t, rho)

    while not W2_check(f, x, d, descent, t, rho):  # Refining
        t = (t_plus + t_minus) / 2
        if W1_check(f, x, d, descent, t, sigma):
            t_minus = t
        else:
            t_plus = t

    if verbose:
        xt = x + t * d
        fxt = f.objective(xt)
        gradxt = f.gradient(xt)
        print('WolfePowellSearch terminated with t=', t)
        print('Wolfe-Powell: ', fxt, '<=', fx + t * sigma * descent, ' and ', gradxt.T @ d, '>=', rho * descent)

    return t
