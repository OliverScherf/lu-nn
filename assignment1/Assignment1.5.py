'''
Created on 05.03.2018

@author: Group 35
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

xMatrix = np.array([[0, 0, 1, 1],
                    [0, 1, 0, 1]])
xorVector = np.array([0, 1, 1, 0])

def sigmoid(n):
    return 1.0/(1.0+np.exp(-n))

def hyperbolicTangent(n):
    return np.tanh(n)

def linearRectifier(n):
    return np.log(1 + np.exp(n))

def activationValue(x1,x2, weights):
    global activationFunc
    return activationFunc(np.sum([x1, x2] * weights[0:2]) + weights[2])
        
def xor_net(x1, x2, weights):
    hidden1 = activationValue(x1, x2, weights[0:3])
    hidden2 = activationValue(x1, x2, weights[3:6])
    output = activationValue(hidden1, hidden2, weights[6:9])
    return output
    

def squaredError(x1, x2, d, weights):    
    return (xor_net(x1, x2, weights) - d) ** 2

def grdmse(weights):
    # approximate derivation
    e = 10**-3
    newWeights = np.zeros(9)
    for j in range(0, 9):
        copiedWeights = copy.copy(weights)
        copiedWeights[j] += e
        newWeights[j]   = ((mse(copiedWeights) - mse(weights)) / e)
    return newWeights

def mse(weights):
    error = 0.0
    for i in range(0, len(xorVector)):
        error += squaredError(xMatrix[0][i], xMatrix[1][i], xorVector[i], weights)
    return error

def getMisclassifiedInputAmount(weights):
    misclassified = 0
    for i in range(0, len(xorVector)): 
        classified = 0
        if (xor_net(xMatrix[0][i], xMatrix[1][i], weights) < 0.5):
            classified = 0
        else:
            classified = 1
        if (xorVector[i] != classified):
            misclassified += 1
            
    return misclassified


def calculateMseAndMisclassifiedInputs(weights, ac, learningRate):
    # we need to set the activation function here (it will be used in the activationValue function)
    global activationFunc
    activationFunc = ac
    achievedMse = []
    misclassifiedInputs = []
    print("Initial Error: ", mse(weights))
    for i in range(0, 2000):
        weights = weights - learningRate * grdmse(weights)
        achievedMse.append(mse(weights))
        misclassifiedInputs.append(getMisclassifiedInputAmount(weights))
    print("Improved Error: ", mse(weights))
    return (achievedMse, misclassifiedInputs)


def evaluateConfiguration(weights, weightInitStrategy, learningRate):
    (sigmoidMSE, sigmoidMisclassified) = calculateMseAndMisclassifiedInputs(copy.copy(weights), sigmoid, learningRate)
    (tangentMSE, tangentMisclassfied) = calculateMseAndMisclassifiedInputs(copy.copy(weights), hyperbolicTangent, learningRate)
    (linRectMSE, linRectMisclassified) = calculateMseAndMisclassifiedInputs(copy.copy(weights), linearRectifier, learningRate)
    plt.figure()
    plt.plot(sigmoidMSE, "-",  label="Sigmoid", alpha=0.5)
    plt.plot(tangentMSE, "-.", label="Hyperbolic Tangent", alpha=0.5)
    plt.plot(linRectMSE, "--", label="Linear Rectifier", alpha=0.5)
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("MSE")
    plt.title("Learning Rate = " + str(learningRate) + ", Weight Initialization Strategy: " + weightInitStrategy)
    plt.savefig("mse_" + weightInitStrategy.replace(" ", "") + "_" + str(learningRate).replace(".", "") + ".png")

    plt.figure()
    plt.plot(sigmoidMisclassified, "-",  label="Sigmoid", alpha=0.5)
    plt.plot(tangentMisclassfied, "-.", label="Hyperbolic Tangent", alpha=0.5)
    plt.plot(linRectMisclassified, "--", label="Linear Rectifier", alpha=0.5)
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Misclassified Inputs")
    plt.title("Learning Rate " + str(learningRate) + ", Weight Initialization Strategy: " + weightInitStrategy)
    plt.savefig("misclassification_" + weightInitStrategy.replace(" ", "") + "_" + str(learningRate).replace(".", "") + ".png")



def main():
    learningRates = [0.1, 0.5, 1.0, 2.0, 4.0]
    weights = np.random.rand(9)
    # weights 2, 5, 8 is the bias, thats why we set it to 1.
    weights[2] = 1
    weights[5] = 1
    weights[8] = 1
    for lr in learningRates:
        evaluateConfiguration(weights, "random", lr)
        
    weights = np.ones(9)
    for lr in learningRates:
        evaluateConfiguration(weights, "all 1", lr)
      
    weights = np.full(9, 0.5)
    weights[2] = 1
    weights[5] = 1
    weights[8] = 1
    for lr in learningRates:
        evaluateConfiguration(weights, "all 0.5", lr)
        
    
    weights = np.zeros(9)
    weights[2] = 1
    weights[5] = 1
    weights[8] = 1
    for lr in learningRates:
        evaluateConfiguration(weights, "all 0", lr)
        
    
if __name__ == "__main__":
    main()