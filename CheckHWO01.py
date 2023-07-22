# Optimization for Engineers - Dr.Johannes Hild
# Programming Homework Check Script
# Do not change this file

print('Welcome to Optimization for Engineers.\n')
print('If this script fails, then your programming homework is not working correctly.')

import numpy as np
import simpleValleyObjective as SO
import nonlinearObjective as NO
import CGSolver as CG
import WolfePowellSearch as WP

A = np.array([[4, 1, 0], [1, 7, 0], [ 0, 0, 3]], dtype=float)
b = np.array([[5], [8], [3]], dtype=float)
delta = 1.0e-6
x = CG.CGSolver(A, b, delta, 1)
xe = np.array([[1], [1], [1]])
if np.linalg.norm(x - xe) > 1.0e-3:
    raise Exception('Your CGSolver is not working correctly.')
else:
    print('Check 01 okay')

A = np.array([[484, 374, 286, 176, 88], [374, 458, 195, 84, 3], [286, 195, 462, -7, -6], [176, 84, -7, 453, -10], [88, 3, -6, -10, 443]], dtype=float)
b = np.array([[1320], [773], [1192], [132], [1405]], dtype=float)
x = CG.CGSolver(A, b, delta, 1)
xe = np.array([[1], [0], [2], [0], [3]])

if np.linalg.norm(x - xe) > 1.0e-3:
    raise Exception('Your CGSolver is not working correctly for other dimensions.')
else:
    print('Check 02 okay')

p = np.array([[0], [1]])
myObjective = SO.simpleValleyObjective(p)
x = np.array([[-1.01], [1]])
d = np.array([[1], [1]])
sigma = 1.0e-3
rho = 1.0e-2
t = WP.WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
te = 1
if t != te:
    raise Exception('Your Wolfe-Powell search is not recognizing t = 1 as valid starting point.')
else:
    print('Check 03 okay')

x = np.array([[-1.2], [1]])
d = np.array([[0.1], [1]])
t = WP.WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
te = 16
if t != te:
    raise Exception('Your Wolfe-Powell search is not fronttracking correctly.')
else:
    print('Check 04 okay')

x = np.array([[-0.2], [1]])
d = np.array([[1], [1]])
t = WP.WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
te = 0.25
if t != te:
    raise Exception('Your Wolfe-Powell search is not refining correctly.')
else:
    print('Check 05 okay')

myObjective = NO.nonlinearObjective()
x = np.array([[0.53], [-0.29]])
d = np.array([[-3.88], [1.43]])
t = WP.WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
te = 0.0938
if np.abs(t - te) > 1.0e-3:
    raise Exception('Your Wolfe-Powell search is not working for general objective class.')
else:
    print('Check 06 okay')

if CG.matrnr() == 0:
    raise Exception('Please set your matriculation number in CGSolver.py!')
elif WP.matrnr() == 0:
    raise Exception('Please set your matriculation number in WolfePowellSearch.py!')
else:
    print('Everything seems to be fine, please return your files in StudOn')
