# Optimization for Engineers - Dr.Johannes Hild
# Mock Homework to check setup
# Do not change this file

import noisyObjective as NOI
import projectionInBox as PB
import numpy as np

print('Welcome to Optimization for Engineers.\n')
print('If this script fails, then your setup is not working correctly.')
print('First we check if the math package numpy is installed.')

X = np.power(2, 3)
Y = 2**3
if X == Y:
    print('numpy seems to work.\n')

print('Next we check if the function definitions in noisyObjective.py are available.')

p = np.array([[3]])
myObjective = NOI.noisyObjective(p)


print('We call noisy objective now several times,')
print('but because of the noise we get slightly different results for each call.')
x = np.array([[0], [0], [0], [0], [0], [0], [0], [0]])
for i in range(5):
    f = myObjective.objective(x)
    print('The noisy objective returns at x=[0,0,0,0,0,0,0,0]^t')
    print(f)

print('Next we check if the function definitions in projectionInBox.py are available.')

a = np.array([[1], [1], [1], [1], [1], [1], [1], [1]])
b = a.copy()+1


print('Now we build box constraints Omega=[1,2]^8')
P = PB.projectionInBox(a, b)
print('Projecting x=[0,0,0,0,0,0,0,0]^t onto Omega leads to')
xp = P.project(x)
print(xp, '\n')

print('\nEverything seems to be fine, please return the files in StudOn for training')
