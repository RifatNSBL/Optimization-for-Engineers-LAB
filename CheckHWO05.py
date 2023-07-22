# Optimization for Engineers - Dr.Johannes Hild
# Programming Homework Check Script
# Do not change this file

print('Welcome to Optimization for Engineers.\n')
print('If this script fails, then your programming homework is not working correctly.')

import numpy as np
import SUCSimplexGradient as SUC
import nonlinearObjective as NO
import multidimensionalObjective as MO
import noisyObjective as NOI
import implicitFiltering as IF

myObjective = MO.multidimensionalObjective()
x = np.array([[1], [1], [1], [1], [1], [1], [1], [1]], dtype=float)
h = 1.0e-6
g = SUC.SUCSimplexGradient(myObjective, x, h)
ge = myObjective.gradient(x)
if np.linalg.norm(g-ge) > 1.0e-3:
    raise Exception('Your SUCSimplexGradient returns a wrong gradient')
else:
    print('Check 01 is okay')

myObjective = NO.nonlinearObjective()
x = np.array([[1], [1]], dtype=float)
h = 1.0e-6
g = SUC.SUCSimplexGradient(myObjective, x, h)
ge = myObjective.gradient(x)
if np.linalg.norm(g-ge) > 1.0e-3:
    raise Exception('Your SUCSimplexGradient returns a wrong gradient')
else:
    print('Check 02 is okay')

myObjective = NO.nonlinearObjective()
x = np.array([[0.261], [-0.209]], dtype=float)
h = 1.0e-2
g1 = SUC.SUCStencilFailure(myObjective, x, h)
x = np.array([[1], [1]], dtype=float)
g2 = SUC.SUCStencilFailure(myObjective, x, h)
if (not g1) or g2:
    raise Exception('Your SUCSimplexGradient returns a wrong stencil failure check')
else:
    print('Check 03 is okay')

myObjective = MO.multidimensionalObjective()
x0 = np.array([[1],[1],[1],[1],[1],[1],[1],[1]], dtype=float)
h = np.array([[1], [0.1], [0.01], [0.001], [0.0001], [0.00001]], dtype=float)
xmin = IF.implicitFiltering(myObjective, x0, h, 1.0e-1, 1)
xe = np.array([[1.027],[0],[0],[0],[0],[0],[0],[0]], dtype=float)
if np.linalg.norm(xmin-xe) > 1.0e2:
    raise Exception('Your implicitFiltering returns a wrong LMP for the noise free objective')
else:
    print('Check 04 is okay.')

myObjective = NOI.noisyObjective()
x0 = np.array([[10],[1],[1],[1],[1],[1],[1],[1]], dtype=float)
h = np.array([[1], [0.1], [0.01], [0.001], [0.0001], [0.00001]], dtype=float)
xmin = IF.implicitFiltering(myObjective, x0, h, 1.0e-1, 1)
xe = np.array([[1.027],[0],[0],[0],[0],[0],[0],[0]], dtype=float)
if np.linalg.norm(xmin-xe) > 5.0e-2:
    raise Exception('Your implicitFiltering returns a wrong LMP for the noisy problem. ')
else:
    print('Check 05 is okay. You also minimized the noisy problem!')

if SUC.matrnr() == 0:
    raise Exception('Please set your matriculation number in SUCSimplexGradient.py!')
elif IF.matrnr() == 0:
    raise Exception('Please set your matriculation number in implicitFiltering.py!')
else:
    print('Everything seems to be fine, please return your files in StudOn')
