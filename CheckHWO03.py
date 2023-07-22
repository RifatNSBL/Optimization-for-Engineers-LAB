# Optimization for Engineers - Dr.Johannes Hild
# Programming Homework Check Script
# Do not change this file

print('Welcome to Optimization for Engineers.\n')
print('If this script fails, then your programming homework is not working correctly.')

import numpy as np
import simpleValleyObjective as SO
import quadraticObjective as QO
import nonlinearObjective as NO
import bananaValleyObjective as BO
import projectionInBox as PB
import projectedBacktrackingSearch as PS
import projectedBFGSDescent as PD

p = np.array([[0], [1]])
myObjective = SO.simpleValleyObjective(p)
a = np.array([[-1], [-1]])
b = np.array([[2], [2]])
myBox = PB.projectionInBox(a, b)
x = np.array([[-1.01], [1]])
d = np.array([[1], [1]])
sigma = 1.0e-3
t = PS.projectedBacktrackingSearch(myObjective, myBox, x, d, sigma, 1)
te = 1
if t != te:
    raise Exception('Your projected backtracking search is not recognizing t = 1 as valid starting point.')
else:
    print('Check 01 okay')

p = np.array([[0], [1]])
myObjective = SO.simpleValleyObjective(p)
a = np.array([[-2], [1]])
b = np.array([[2], [2]])
myBox = PB.projectionInBox(a, b)
x = np.array([[1], [1]])
d = np.array([[-1.99], [0]])
sigma = 0.5
t = PS.projectedBacktrackingSearch(myObjective, myBox, x, d, sigma, 1)
te = 0.5
if t != te:
    raise Exception('Your projected backtracking search is not backtracking correctly.')
else:
    print('Check 02 okay')

p = np.array([[1], [1]])
myObjective = SO.simpleValleyObjective(p)
a = np.array([[1], [1]])
b = np.array([[2], [2]])
myBox = PB.projectionInBox(a, b)
x0 = np.array([[2], [2]], dtype=float)
eps = 1.0e-3
xmin = PD.projectedBFGSDescent(myObjective, myBox, x0, eps, 1)
xe = np.array([[1], [1]], dtype=float)
if np.linalg.norm(xmin-xe) > 1.0e-2:
    raise Exception('Your projected BFGS descent is not working.')
else:
    print('Check 03 is okay')

A = np.eye(3)
B = np.array([[3],[5],[7]], dtype=float)
C = 1
myObjective = QO.quadraticObjective(A, B, C)
aa = np.array([[1], [1], [1]])
bb = np.array([[2], [2], [2]])
myBox = PB.projectionInBox(aa, bb)
x0 = np.array([[1], [2], [3]], dtype=float)
eps = 1.0e-3
xmin = PD.projectedBFGSDescent(myObjective, myBox, x0, eps, 1)
xe = np.array([[1], [1], [1]], dtype=float)
if np.linalg.norm(xmin - xe) > 1.0e-2:
    raise Exception('Your projected BFGS descent is not working for general dimensions.')
else:
    print('Check 04 is okay')

myObjective = NO.nonlinearObjective()
x0 = np.array([[0.15], [0.17]], dtype=float)
xmin = PD.projectedBFGSDescent(myObjective, myBox, x0, eps, 1)
xe = np.array([[1], [1]], dtype=float)
if np.linalg.norm(xmin-xe) > 1.0e-2:
    raise Exception('Your projected BFGS descent is not working for general objective class.')
else:
    print('Check 05 is okay')

a = np.array([[-2], [-2]])
myBox = PB.projectionInBox(a, b)
x0 = np.array([[1.5], [2]], dtype=float)
xmin = PD.projectedBFGSDescent(myObjective, myBox, x0, eps, 1)
xe = np.array([[0.26],[-0.21]], dtype=float)
if np.linalg.norm(xmin-xe) > 1.0e-2:
    raise Exception('Your projected BFGS descent is not working, maybe your reduction is done wrongly?')
else:
    print('Check 06 is okay')

myObjective = BO.bananaValleyObjective()
a = np.array([[-10], [-10]])
b = np.array([[10], [10]])
myBox = PB.projectionInBox(a, b)
x0 = np.array([[0], [1]], dtype=float)
eps = 1.0e-6
xmin = PD.projectedBFGSDescent(myObjective, myBox, x0, eps, 1)
xe = np.array([[1], [1]], dtype=float)
if np.linalg.norm(xmin-xe) > 1.0e-2:
    raise Exception('Your projected Newton descent is not working.')
else:
    print('Check 07 is okay if the number of iterations is less than 30. Otherwise maybe your hessian is reduced wrongly.')

if PS.matrnr() == 0:
    raise Exception('Please set your matriculation number in projectedBacktrackingSearch.py!')
elif PD.matrnr() == 0:
    raise Exception('Please set your matriculation number in projectedBFGSDescent.py!')
else:
    print('Everything seems to be fine, please return your files in StudOn')
