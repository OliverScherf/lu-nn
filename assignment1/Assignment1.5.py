'''
Created on 05.03.2018

@author: Matthias Müller-Brockhausen & Oliver Scherf
'''
import numpy as np
import sys

import itertools
import math
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import copy

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib.pyplot import plot
from Utils import plot_confusion_matrix
from numpy.matlib import rand

global activationFunc

def sigmoid(n):
    return 1.0/(1.0+np.exp(-n))

def hyperbolicTangent(n):
    return np.tanh(n)

def linearRectifier(n):
    return np.log(1 + np.exp(n))
    # max(0, n) does'nt work, so we just use the "smooth approximation"
    #if n < 0.0:
    #    return 0
    #else:
    #    return n

def activationValue(x1,x2, weights):
    global activationFunc
    return activationFunc(np.sum([x1, x2] * weights[0:2]) + weights[2])
        
def xor_net(x1, x2, weights):
    hidden1 = activationValue(x1, x2, weights[0:3])
    hidden2 = activationValue(x1, x2, weights[3:6])
    output = activationValue(hidden1, hidden2, weights[6:9])
    return output
    

def calculateError(x1, x2, d, weights):    
    return (xor_net(x1, x2, weights) - d) ** 2

def grdmse(weights):
    xMatrix = np.array([[0, 0, 1, 1],
                        [0, 1, 0, 1]])
    xorVector = np.array([0, 1, 1, 0])
    e = 10**-3
    
    
    returnVector = np.zeros(9)
    for j in range(0, 9):
        sliceWeights0 = copy.copy(weights)
        sliceWeights0[j] += e
        returnVector[j]   = ((mse(sliceWeights0) - mse(weights)) / e)
    return returnVector

def mse(weights):
    xMatrix = np.array([[0, 0, 1, 1],
                        [0, 1, 0, 1]])
    xorVector = np.array([0, 1, 1, 0])
    
    error = 0.0
    for i in range(0, len(xorVector)):
        error += calculateError(xMatrix[0][i], xMatrix[1][i], xorVector[i], weights)
    return error


def getMseFor(weights, ac, learningRate):
    global activationFunc
    activationFunc = ac
    achievedMse = []
    print("Initial Error: ", mse(weights))
    achievedMse.append(mse(weights))
    for i in range(0, 2000):
        weights = weights - learningRate * grdmse(weights)
        achievedMse.append(mse(weights))
    return achievedMse


def plotMseFor(weights, weightName, learningRate):
    plt.figure()
    plt.plot(getMseFor(copy.copy(weights), sigmoid, learningRate), "-", label="Sigmoid", alpha=0.5)
    plt.plot(getMseFor(copy.copy(weights), hyperbolicTangent, learningRate), "-.", label="Hyperbolic Tangent", alpha=0.5)
    plt.plot(getMseFor(copy.copy(weights), linearRectifier, learningRate), "--", label="Linear Rectifier", alpha=0.5)
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("MSE")
    plt.title("Learning Rate " + str(learningRate) + ", Weight Initialization Strategy: " + weightName)
    plt.savefig("mse_" + weightName + "_" + str(learningRate) + ".png")


def main():
    weights = np.random.rand(9)
    weights[2] = 1
    weights[5] = 1
    weights[8] = 1
    plotMseFor(weights, "random except biasses are 1", 0.1)
    plotMseFor(weights, "random except biasses are 1", 0.5)
    plotMseFor(weights, "random except biasses are 1", 1)
    plotMseFor(weights, "random except biasses are 1", 2)
    weights = np.ones(9)
    plotMseFor(weights, "all 1", 0.1)
    plotMseFor(weights, "all 1", 0.5)
    plotMseFor(weights, "all 1", 1)
    plotMseFor(weights, "all 1", 2)
    weights = np.full(9, 0.5)
    weights[2] = 1
    weights[5] = 1
    weights[8] = 1
    plotMseFor(weights, "all 0.5 except biasses are 1", 0.1)
    plotMseFor(weights, "all 0.5 except biasses are 1", 0.5)
    plotMseFor(weights, "all 0.5 except biasses are 1", 1)
    plotMseFor(weights, "all 0.5 except biasses are 1", 2)
    weights = np.zeros(9)
    weights[2] = 1
    weights[5] = 1
    weights[8] = 1
    plotMseFor(weights, "all 0 except biasses are 1", 0.1)
    plotMseFor(weights, "all 0 except biasses are 1", 0.5)
    plotMseFor(weights, "all 0 except biasses are 1", 1)
    plotMseFor(weights, "all 0 except biasses are 1", 2)
    
    
if __name__ == "__main__":
    main()