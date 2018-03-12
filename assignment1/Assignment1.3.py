'''
Created on 24.02.2018

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

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

def extractFeature(img):
    upper = img[0:128] + 1
    lower = img[128:256] + 1
    ratio = np.sum(upper) / np.sum(lower)
    return ratio

def filterSet(setIn, setOut, values):
    filteredSetIn = []
    filteredSetOut = []
    for i in range(0, len(setIn)):
        if (np.any(setOut[i] == values)):
            filteredSetIn.append(setIn[i])
            filteredSetOut.append(setOut[i])
    return (filteredSetIn, filteredSetOut)

def classify(img, b1, b2, x_axis):
    res = extractFeature(img)
    idx = find_nearest(x_axis, res)
    print(idx)
    
    
    #print(idx, b1[idx], b2[idx])

    if (b1[idx] > b2[idx]):
        return 1
    else:
        return 8

def main():
    trainIn = np.genfromtxt('data/train_in.csv', delimiter=',')
    trainOut = np.genfromtxt('data/train_out.csv', delimiter=',')
    testIn = np.genfromtxt('data/test_in.csv', delimiter=',')
    testOut = np.genfromtxt('data/test_out.csv', delimiter=',')
    filteredSetIn, filteredSetOut = filterSet(trainIn, trainOut, [1, 8])
    
    
    STEPS = 15
    
    features1 = []
    features8 = []
    for i in range(0, len(filteredSetIn)):
        result = extractFeature(filteredSetIn[i])
        
        if (filteredSetOut[i] == 1):
            features1.append(result)
        else:
            features8.append(result)

    plt.figure()
    hist1 = np.histogram(features1, bins=STEPS)
    hist8 = np.histogram(features8, bins=STEPS)
    aa = plt.hist(features1, bins=STEPS, label="#1")
    bb = plt.hist(features8, bins=STEPS, label="#8")
    plt.xlabel("Amount of occurences")
    plt.ylabel("Classified value")
    plt.legend()
    plt.savefig("Histogram.png")
    print(hist1)
    print(len(hist1[0]))
    
    
    numSamples1 = np.sum(hist1[0])
    numSamples8 = np.sum(hist8[0])
    numSamples = numSamples1 + numSamples8    

    aPrio1 = numSamples1 / numSamples
    aPrio8 = numSamples8 / numSamples

    classProbabilty1 = []
    classProbabilty8 = []
    for i in range(0, len(hist1[0])):
        classProbabilty1.append(hist1[0][i] / numSamples1)
        classProbabilty8.append(hist8[0][i] / numSamples8)
    
    bayesProbabilty1 = []
    bayesProbabilty8 = []
    
    
    for i in range(0, len(hist1[0])):
        px = classProbabilty1[i] * aPrio1 + classProbabilty8[i] * aPrio8
        # if there are no values for that extracted feature just assume 0.5
        if (px == 0.0):
            bayesProbabilty1.append(0.5)
            bayesProbabilty8.append(0.5)
        else: 
            bayesProbabilty1.append(classProbabilty1[i] * aPrio1 / px)
            bayesProbabilty8.append(classProbabilty8[i] * aPrio8 / px)
    
    min = np.round((np.amin([np.amin(hist1[1]), np.amin(hist8[1])]) - 0.1))
    max = np.round(np.amax([np.amax(hist1[1]), np.amax(hist8[1])]) + 0.1)
    x_axis = np.linspace(min, max, STEPS)
    
    plt.figure()
    plt.plot(x_axis, bayesProbabilty1)
    plt.plot(x_axis, bayesProbabilty8)
    plt.savefig("Bayes for 1 and 8.png")
    
   
    testIn, testOut = filterSet(testIn, testOut, [1, 8])
    print("Bayes1 Length:", len(bayesProbabilty1))
    correct = 0
    total = 0
    for i in range(0, len(testOut)):
        total += 1
        classified = classify(testIn[i], bayesProbabilty1, bayesProbabilty8, x_axis)
        if (classified == testOut[i]):
            correct += 1
        print("Actual: ", testOut[i], "Classified",classify(testIn[i], bayesProbabilty1, bayesProbabilty8, x_axis))
    
    print("Correct", correct, "Total", total, "Ratio:", correct/total)
    
if __name__ == "__main__":
    main()