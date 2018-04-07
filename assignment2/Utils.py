'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import numpy as np
import sys

def maxIndex(vector):
    maxValue = -sys.maxsize - 1
    idx = -1
    for i in range(0, len(vector)):
        if (vector[i] > maxValue):
            maxValue = vector[i]
            idx = i
    
    return idx

def findMostMisclassifiedDigits(model, x_test, y_test):
    predicted = model.predict(x_test, batch_size=128)
    
    print("predicated", predicted)
    np.savetxt("Predicted.csv", predicted, delimiter=",", fmt='%2.2f')
    print(len(x_test))
    
    misclassified = np.zeros(10)
    for i in range(0, len(predicted)):
        regocnized = maxIndex(predicted[i])
        actual = maxIndex(y_test[i]) #we need to find the maxIndex for the y_test variable because keras.to_cateogrial was called
        if (regocnized != actual):
            misclassified[actual] += 1
    
    print(misclassified)
    
def permutateData(x, seed):
    for i in range(0, len(x)):
        np.random.seed(seed)
        x[i] = np.random.permutation(x[i])

    return x