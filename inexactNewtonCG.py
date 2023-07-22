# Optimization for Engineers - Dr.Johannes Hild
# inexact Newton descent

# Purpose: Find xmin to satisfy norm(gradf(xmin))<=eps
# Iteration: x_k = x_k + t_k * d_k
# d_k starts as a steepest descent step and then CG steps are used to improve the descent direction until negative curvature is detected or a full Newton step is made.
# t_k results from Wolfe-Powell

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# dH = directionalHessApprox(f, x, d) from directionalHessApprox.py
# t = WolfePowellSearch(f, x, d) from WolfePowellSearch.py

# Test cases:
# myObjective = nonlinearObjective()
# x0 = np.array([[-0.01], [0.01]])
# eps = 1.0e-6
# xmin = inexactNewtonCG(myObjective, x0, eps, 1)
# should return
# xmin close to [[0.26],[-0.21]]

# myObjective = nonlinearObjective()
# x0 = np.array([[-0.6], [0.6]])
# eps = 1.0e-3
# xmin = inexactNewtonCG(myObjective, x0, eps, 1)
# should return
# xmin close to [[-0.26],[0.21]]

# myObjective = nonlinearObjective()
# x0 = np.array([[0.6], [-0.6]])
# eps = 1.0e-3
# xmin = inexactNewtonCG(myObjective, x0, eps, 1)
# should return
# xmin close to [[-0.26],[0.21]]


import numpy as np
import WolfePowellSearch as WP
import directionalHessApprox as DHA

def matrnr():
    # set your matriculation number here
    matrnr = 23184261
    return matrnr


def inexactNewtonCG(f, x0: np.array, eps=1.0e-3, verbose=0):

    if eps <= 0:
        raise TypeError('range of eps is wrong!')

    if verbose:
        print('Start inexactNewtonCG...')

    countIter = 0
    x = x0.copy()
    gradx = f.gradient(x)
    gradx_norm = np.linalg.norm(gradx, 2)
    eta = np.min([0.5, np.sqrt(gradx_norm) * gradx_norm])

    while gradx_norm > eps:
        d = -f.gradient(x)
        dH = DHA.directionalHessApprox(f, x, d)
        ro = np.dot(d.T, dH)

        if ro > (eps * np.linalg.norm(d, 2) ** 2):
            r_j = f.gradient(x)
            d_j = np.copy(-r_j)
            x_j = np.copy(x)
            dA = np.copy(dH)
            ro_j = ro
            t_j = (np.linalg.norm(r_j, 2) ** 2) / ro_j
            x_j = x_j + t_j * d_j
            r_old = np.copy(r_j)
            r_j = r_old + t_j * dA
            betta_j = (np.linalg.norm(r_j, 2) ** 2) / (np.linalg.norm(r_old, 2) ** 2)
            d_j = -r_j + betta_j * d_j

            while np.linalg.norm(r_j) > eta:
                dA = DHA.directionalHessApprox(f, x, d_j)
                ro_j = np.dot(d_j.T, dA)
                t_j = np.linalg.norm(r_j, 2) ** 2 / ro_j
                x_j = x_j + t_j * d_j
                r_old = np.copy(r_j)
                r_j = r_old + t_j * dA
                betta_j = (np.linalg.norm(r_j) ** 2) / (np.linalg.norm(r_old) ** 2)
                d_j = -r_j + betta_j * d_j
            d = x_j - x

        t_k = WP.WolfePowellSearch(f, x, d)
        x = x + t_k * d
        gradx = f.gradient(x)
        gradx_norm = np.linalg.norm(gradx, 2)
        eta = np.min([0.5, np.sqrt(gradx_norm) * gradx_norm])
        countIter += 1


    if verbose:
        gradx = f.gradient(x)
        print('inexactNewtonCG terminated after ', countIter, ' steps with norm of gradient =', np.linalg.norm(gradx))

    return x
