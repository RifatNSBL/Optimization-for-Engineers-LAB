# Optimization for Engineers - Dr.Johannes Hild
# Programming Homework Check Script
# Do not change this file

print('Welcome to Optimization for Engineers.\n')
print('If this script fails, then your programming homework is not working correctly.')

import numpy as np
import simpleValleyObjective as SO
import nonlinearObjective as NO
import directionalHessApprox as DHA
import inexactNewtonCG as INCG
import multidimensionalObjective as MO

p = np.array([[0], [1]])
myObjective = SO.simpleValleyObjective(p)
x = np.array([[-1.01], [1]])
d = np.array([[1], [1]])
dH = DHA.directionalHessApprox(myObjective, x, d, 1.0e-6, 1)
dH1 = np.array([[1.55491], [0]])

if np.linalg.norm(dH-dH1) > 1.0e-6:
    raise Exception('Your directionalHessApprox is not working correctly.')
else:
    print('Check 01 okay')

myObjective = NO.nonlinearObjective()
x = np.array([[-0.015793], [0.012647]])
d = np.array([[2], [2]])
dH = DHA.directionalHessApprox(myObjective, x, d, 1.0e-6, 1)
dH1 = myObjective.hessian(x)@d

if np.linalg.norm(dH-dH1) > 1.0e-6:
    raise Exception('Your directionalHessApprox is not working correctly.')
else:
    print('Check 02 okay')

x0 = np.array([[-0.01], [0.01]])
eps = 1.0e-6
xmin = INCG.inexactNewtonCG(myObjective, x0, eps, 1)
xe = np.array([[0.26], [-0.21]])
if np.linalg.norm(xmin-xe) > 1.0e-2:
    raise Exception('Your inexactNewtonCG is not working correctly.')
else:
    print('Check 03 okay')

x0 = np.array([[-0.6], [0.6]])
xmin = INCG.inexactNewtonCG(myObjective, x0, eps, 1)
xe = np.array([[0.26], [-0.21]])
if np.linalg.norm(xmin-xe) > 1.0e-2:
    raise Exception('Your inexactNewtonCG walks a wrong path, maybe you switch to steepest descent too often?')
else:
    print('Check 04 okay')

x0 = np.array([[0.6], [-0.6]])
xmin = INCG.inexactNewtonCG(myObjective, x0, eps, 1)
xe = np.array([[-0.26], [0.21]])
if np.linalg.norm(xmin - xe) > 1.0e-2:
    raise Exception('Your inexactNewtonCG walks a wrong path, maybe you make mistakes in choosing the descent directions?')
else:
    print('Check 05 okay')

myObjective = MO.multidimensionalObjective()
x0 = np.array([[1],[1],[1],[1],[1],[1],[1],[1]])
xmin = INCG.inexactNewtonCG(myObjective, x0, eps, 1)
xe = np.array([[1.02614],[0],[0],[0],[0],[0],[0],[0]])
if np.linalg.norm(xmin - xe) > 1.0e-2:
    raise Exception('Your inexactNewtonCG does not work for the 8-dimensional test function?')
else:
    print('Check 06 is okay, if the number of iterations is smaller than 10. Otherwise you use zigzagging directions.')

if DHA.matrnr() == 0:
    raise Exception('Please set your matriculation number in directionalHessApprox.py!')
elif INCG.matrnr() == 0:
    raise Exception('Please set your matriculation number in inexactNewtonCG.py!')
else:
    print('Everything seems to be fine, please return your files in StudOn')
