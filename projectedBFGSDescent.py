# Optimization for Engineers - Dr.Johannes Hild
# projected BFGS descent

# Purpose: Find xmin to satisfy norm(xmin - P(xmin - gradf(xmin)))<=eps
# Iteration: x_k = P(x_k + t_k * d_k)
# d_k is the BFGS direction. If a descent direction check fails, d_k is set to steepest descent and the inverse BFGS matrix is reset.
# t_k results from projected backtracking

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# P: box projection class with method .project() and .activeIndexSet()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# t = projectedBacktrackingSearch(f, P, x, d) from projectedBacktrackingSearch.py

# Test cases:
# p = np.array([[1], [1]])
# myObjective = simpleValleyObjective(p)
# a = np.array([[1], [1]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[2], [2]], dtype=float)
# eps = 1.0e-3
# xmin = projectedBFGSDescent(myObjective, myBox, x0, eps, 1)
# should return xmin close to [[1],[1]]

# myObjective = nonlinearObjective()
# a = np.array([[1], [1]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[0.1], [0.1]], dtype=float)
# eps = 1.0e-3
# xmin = projectedBFGSDescent(myObjective, myBox, x0, eps, 1)
# should return xmin close to [[1],[1]]

# myObjective = nonlinearObjective()
# a = np.array([[-2], [-2]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[1.5], [2]], dtype=float)
# eps = 1.0e-3
# xmin = projectedBFGSDescent(myObjective, myBox, x0, eps, 1)
# should return xmin close to [[-0.26],[0.21]] (if it is close to [[0.26],[-0.21]] then maybe your reduction is done wrongly)

# myObjective = bananaValleyObjective()
# a = np.array([[-10], [-10]])
# b = np.array([[10], [10]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[0], [1]], dtype=float)
# eps = 1.0e-6
# xmin = projectedBFGSDescent(myObjective, myBox, x0, eps, 1)
# should return xmin close to [[1],[1]] in less than 30 iterations. If you have too much iterations, then maybe the hessian is used wrongly.


import numpy as np
import projectedBacktrackingSearch as PB


def matrnr():
    # set your matriculation number here
    matrnr = 23184261
    return matrnr


def projectedBFGSDescent(f, P, x0: np.array, eps=1.0e-3, verbose=0):
    if eps <= 0:
        raise TypeError('range of eps is wrong!')

    if verbose:
        print('Start projectedBFGSDescent...')

    countIter = 0
    xp = P.project(x0)
    gradx = f.gradient(xp)
    # INCOMPLETE CODE
    B_k = np.eye(x0.shape[0])
    condition = np.linalg.norm(x0 - P.project(x0 - f.gradient(x0)))
    while condition > eps:
        list = P.activeIndexSet(x0)
        B_k_reduced = np.copy(B_k)
        B_k_reduced[list, :] = 0
        B_k_reduced[:, list] = 0
        for i in list:
            B_k_reduced[i, i] = 1
        d_k = - B_k_reduced @ f.gradient(xp)
        if f.gradient(xp).T @ d_k >= 0:
            B_k = np.eye(xp.shape[0])
            d_k = - gradx
        t = PB.projectedBacktrackingSearch(f, P, xp, d_k)
        x_old = xp.copy()
        xp = P.project(xp + t * d_k)
        dx = xp - x_old
        dg = f.gradient(xp) - f.gradient(x_old)
        r = dx - (B_k @ dg)
        B_k += (r @ dx.T + dx @ r.T) / (dg.T @ dx) - ((r.T @ dg) * (dx @ dx.T) / (dg.T @ dx) ** 2)
        condition = np.linalg.norm(xp - P.project(xp - f.gradient(xp)))
        gradx = f.gradient(xp)
        countIter += 1

    if verbose:
        print('projectedBFGSDescent terminated after ', countIter, ' steps with norm of stationarity =',
              np.linalg.norm(xp - P.project(xp - gradx)))

    return xp
