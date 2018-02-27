'''
Created on 27.02.2018

@author: Matthias Müller-Brockhausen & Oliver Scherf
'''
import numpy as np
import sys

import itertools
import math
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib.pyplot import plot

def maxIndex(vector):
    max = -sys.maxsize - 1
    idx = -1
    for i in range(0, len(vector)):
        if (vector[i] > max):
            max = vector[i]
            idx = i
    
    return idx

def classify(img, weights, bias):
    y = np.zeros(10)
    for i in range(0, len(y)):
        y[i] = np.sum(img * weights[i]) + bias[i]
    return y    
    
    
def classifyAll(trainIn, weights, bias):
    classified = np.zeros((len(trainIn), 10))
    for i in range(0, len(trainIn)):
        classified[i] = classify(trainIn[i], weights, bias)
    return classified

def updateWeights(img, weights, bias, classified, desired):
    desired = int(desired)
    compare = classified[desired]
    for i in range(0, len(classified)):
        if (i == desired):
            weights[i] = weights[i] + img
        elif (compare < classified[i]):
            weights[i] = weights[i] - img
            
    return weights

def testWith(setIn, setOut, weights, bias):
    total = 0
    correct = 0
    for i in range(0, len(setIn)):
        total += 1
        classified = maxIndex(classify(setIn[i], weights, bias))
        if (classified == setOut[i]):
            correct += 1
    
    print("Correct", correct, "Total", total, "Ratio:", correct/total)
        

def main():
    trainIn = np.genfromtxt('data/train_in.csv', delimiter=',')
    trainOut = np.genfromtxt('data/train_out.csv', delimiter=',')
    testIn = np.genfromtxt('data/test_in.csv', delimiter=',')
    testOut = np.genfromtxt('data/test_out.csv', delimiter=',')
    
    weights = np.random.rand(10, 256)
    bias = np.zeros(10)
    y = np.zeros(10)
    classified = classifyAll(trainIn, weights, bias)
    
    foundMisclassified = True
    iteration = 0
    while foundMisclassified:
        foundMisclassified = False
        for i in range(0, len(trainIn)):
            if (maxIndex(classified[i]) != trainOut[i]):
                foundMisclassified = True
                weights = updateWeights(trainIn[i], weights, bias, classified[i], trainOut[i])
        classified = classifyAll(trainIn, weights, bias)
        print(iteration)
        iteration += 1
    
    testWith(testIn, testOut, weights, bias)
    
    
    
if __name__ == "__main__":
    main()