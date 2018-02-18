'''
Created on 10.02.2018

@author: Oliver Scherf
'''

import numpy as np
import math

def logisticFunction(n):
    return 1 / (1 + math.exp(-n))

def decision(n):
    if (n < 0.5):
        return 0
    else:
        return 1
    
# Non vectorized solution
def scalar(xMatrix, w, i):
    print("Scalar for column",i, ":")
    neuron1 = logisticFunction(w[0] * xMatrix[0][0 + i] + w[1] * xMatrix[1][0 + i])
    neuron2 = logisticFunction(w[2] * xMatrix[0][0 + i] + w[3] * xMatrix[1][0 + i])
    neuron3 = logisticFunction(w[4] * neuron1 + w[5] * neuron2)
    return decision(neuron3)

# Non vectorized    
def vectorized(xMatrix, w):
    log = np.vectorize(logisticFunction) # Function to apply logisticFunction to every element
    dec = np.vectorize(decision) # Function to apply decision to every element
    
    neuron1 = log((w[0:2].dot(xMatrix)))
    neuron2 = log((w[2:4].dot(xMatrix)))
    tmp = np.column_stack((neuron1,neuron2)).T
    neuron3 = log((w[4:6].dot(tmp)))
    
    result = dec(neuron3)
    return result
    
def runNTimes(xMatrix, solutionVector, N):
    print("Will run", N,"times with solution vector", solutionVector)
    correct = 0
    for x in range(0, N):
        w = np.random.randn(6)
        if (np.all((vectorized(xMatrix, w) == solutionVector))):
            correct = correct + 1
    print("Correct: ", correct)

def main():
    # The feature space
    xMatrix = np.array([[0, 0, 1, 1],
                        [0, 1, 0, 1]])
    
    andVector = np.array([0, 0, 0, 1])
    xorVector = np.array([0, 1, 1, 0])
    runNTimes(xMatrix, xorVector, 10**6)
    runNTimes(xMatrix, andVector, 10**6)

if __name__ == "__main__":
    main()

