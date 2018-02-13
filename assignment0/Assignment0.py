'''
Created on 10.02.2018

@author: Oliver Scherf
'''

import numpy as np
import matplotlib.pyplot as plt

# The feature space
xMatrix = np.array([[0, 0, 1, 1],
                    [0, 1, 0, 1]])
andVector = np.array([0, 0, 0, 1])
xorVector = np.array([0, 1, 1, 0])

featureSpace = plt.figure(1)
plt.imshow(xMatrix)
functions = plt.figure(2)
plt.plot(andVector, "r")
plt.plot(xorVector, "g")
plt.legend(['AND Function', 'XOR Function'])
plt.title('Boolean Functions') 
featureSpace.savefig("feature_space.png")
functions.savefig("functions.png")
plt.show()

weights = np.array([0.5, 0.5])
bias = -0.5484733893874026 #np.random.randn()
y = weights.T.dot(xMatrix) + bias

result = np.heaviside(y, 0)
print('classified: ', result)
print('correct classified: ', result == andVector)





