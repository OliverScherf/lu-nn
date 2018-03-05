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
from Utils import plot_confusion_matrix
from numpy.matlib import rand

STEP_SIZE = 1.0

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
            weights[i] = weights[i] + STEP_SIZE * img
        elif (compare < classified[i]):
            weights[i] = weights[i] - STEP_SIZE * img
            
    return weights

def testWith(setIn, setOut, weights, bias):
    total = 0
    correct = 0
    confusionMatrix = np.zeros((10, 10))
    for i in range(0, len(setIn)):
        total += 1
        recognized = maxIndex(classify(setIn[i], weights, bias))
        
        confusionMatrix[int(setOut[i])][int(recognized)] += 1
        if (recognized == setOut[i]):
            correct += 1
    
    plt.figure()
    plot_confusion_matrix(confusionMatrix, range(0, 10), title="CM for multi class perceptron")
    plt.savefig("cm_multi_class_perceptron.png")
    plt.show()
    print("Correct", correct, "Total", total, "Ratio:", correct/total)

    


def sigmoid(n):
    return 1.0/(1.0+np.exp(-n))

def activationValue(x1,x2, weights):
    return sigmoid(np.sum([x1, x2] * weights[0:2]) + weights[2])
        
def xor_net(x1, x2, weights):
    hidden1 = activationValue(x1, x2, weights[0:2], weights[2])
    hidden2 = activationValue(x1, x2, weights[3:5], weights[5])
    output = activationValue(hidden1, hidden2, weights[6:8], weights[8])
    return output
    

def calculateError(x1, x2, d, weights):    
    return (xor_net(x1, x2, weights) - d) ** 2

def grdmse(weights):
    xMatrix = np.array([[0, 0, 1, 1],
                        [0, 1, 0, 1]])
    xorVector = np.array([0, 1, 1, 0])
    e = 10**-3
    
    for i in range(0, len(xorVector)):
        for j in range(0, 6, 3):
            sliceWeights0 = weights[j:j+3]
            sliceWeights0[0] += e 
            sliceWeights1 = weights[j:j+3]
            sliceWeights1[1] += e 
            sliceWeights2 = weights[j:j+3]
            sliceWeights2[2] += e 

            weights[j]     = (activationValue(xMatrix[0][i], xMatrix[1][i], sliceWeights0) 
                              - activationValue(xMatrix[0][i], xMatrix[1][i], weights[j:j+3])) / e
            
            weights[j + 1]     = (activationValue(xMatrix[0][i], xMatrix[1][i], sliceWeights1) 
                              - activationValue(xMatrix[0][i], xMatrix[1][i], weights[j:j+3])) / e
            
            weights[j + 2]     = (activationValue(xMatrix[0][i], xMatrix[1][i], sliceWeights2) 
                              - activationValue(xMatrix[0][i], xMatrix[1][i], weights[j:j+3])) / e
    
    return weights

def mse(weights):
    xMatrix = np.array([[0, 0, 1, 1],
                        [0, 1, 0, 1]])
    xorVector = np.array([0, 1, 1, 0])
    
    error = 0.0
    for i in range(0, len(xorVector)):
        error += calculateError(xMatrix[0][i], xMatrix[1][i], xorVector[i], weights)
    return error
        

def main():
    weights = np.random.rand(9)
    weights[2] = 1
    weights[5] = 1
    weights[8] = 1
    print(grdmse(weights))
    
    
if __name__ == "__main__":
    main()