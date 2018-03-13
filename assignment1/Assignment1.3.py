'''
Created on 24.02.2018

@author: Group 35
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

def classify(img, b1, b2, scale, digit1, digit2):
    res = extractFeature(img)
    idx = find_nearest(scale, res)
    
    if (b1[idx] > b2[idx]):
        return digit1
    else:
        return digit2

def main():
    trainIn = np.genfromtxt('data/train_in.csv', delimiter=',')
    trainOut = np.genfromtxt('data/train_out.csv', delimiter=',')
    testIn = np.genfromtxt('data/test_in.csv', delimiter=',')
    testOut = np.genfromtxt('data/test_out.csv', delimiter=',')
    
    digit1 = 8
    digit2 = 7
    print("Digits are:", digit1, "and", digit2)
    filteredSetIn, filteredSetOut = filterSet(trainIn, trainOut, [digit1, digit2])
    
    
    STEPS = 15
    
    features1 = []
    features2 = []
    for i in range(0, len(filteredSetIn)):
        result = extractFeature(filteredSetIn[i])
        
        if (filteredSetOut[i] == digit1):
            features1.append(result)
        else:
            features2.append(result)

    plt.figure()
    hist1 = np.histogram(features1, bins=STEPS)
    hist2 = np.histogram(features2, bins=STEPS)
    aa = plt.hist(features1, bins=STEPS, label="#" + str(digit1), alpha=0.7)
    bb = plt.hist(features2, bins=STEPS, label="#" + str(digit2), alpha=0.7)
    plt.xlabel("Classified value")
    plt.ylabel("Amount of occurences")
    plt.legend()
    plt.savefig("Histogram.png")
    
    
    numSamples1 = np.sum(hist1[0])
    numSamples2 = np.sum(hist2[0])
    numSamples = numSamples1 + numSamples2    

    aPrio1 = numSamples1 / numSamples
    aPrio2 = numSamples2 / numSamples

    classProbabilty1 = []
    classProbabilty2 = []
    for i in range(0, len(hist1[0])):
        classProbabilty1.append(hist1[0][i] / numSamples1)
        classProbabilty2.append(hist2[0][i] / numSamples2)
    
    bayesProbabilty1 = []
    bayesProbabilty2 = []
    
    
    for i in range(0, len(hist1[0])):
        px = classProbabilty1[i] * aPrio1 + classProbabilty2[i] * aPrio2
        # if there are no values for that extracted feature just assume 0.5
        if (px == 0.0):
            bayesProbabilty1.append(0.5)
            bayesProbabilty2.append(0.5)
        else: 
            bayesProbabilty1.append(classProbabilty1[i] * aPrio1 / px)
            bayesProbabilty2.append(classProbabilty2[i] * aPrio2 / px)
    
    min = np.round((np.amin([np.amin(hist1[1]), np.amin(hist2[1])]) - 0.1))
    max = np.round(np.amax([np.amax(hist1[1]), np.amax(hist2[1])]) + 0.1)
    x_axis = np.linspace(min, max, STEPS)
    
    plt.figure()
    plt.plot(x_axis, bayesProbabilty1)
    plt.plot(x_axis, bayesProbabilty2)
    plt.savefig("Bayes for " + str(digit1) + " and " + str(digit2) + ".png")
    
   
    testIn, testOut = filterSet(testIn, testOut, [digit1, digit2])
    correct = 0
    total = 0
    for i in range(0, len(testOut)):
        total += 1
        classified = classify(testIn[i], bayesProbabilty1, bayesProbabilty2, x_axis, digit1, digit2)
        if (classified == testOut[i]):
            correct += 1
    
    print("Correct classified:", correct)
    print("Total samples:", total)
    print("Correct/Incorrect ratio:", correct/total)
    
if __name__ == "__main__":
    main()