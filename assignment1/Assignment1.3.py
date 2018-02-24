'''
Created on 24.02.2018

@author: Matthias Müller-Brockhausen & Oliver Scherf
'''
import numpy as np
import sys

import itertools
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib.pyplot import plot

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

def main():
    trainIn = np.genfromtxt('data/train_in.csv', delimiter=',')
    trainOut = np.genfromtxt('data/train_out.csv', delimiter=',')
    testIn = np.genfromtxt('data/test_in.csv', delimiter=',')
    testOut = np.genfromtxt('data/test_out.csv', delimiter=',')
    filteredSetIn, filteredSetOut = filterSet(trainIn, trainOut, [1, 8])
    
    x_axis = np.linspace(0.5, 1.9, 30)
    occurence1 = []
    occurence8 = []
    for i in range(0, len(filteredSetIn)):
        result = extractFeature(filteredSetIn[i])
        idx = np.where(x_axis == result)
        
        if (filteredSetOut[i] == 1):
            occurence1.append(result)
        else:
            occurence8.append(result)

    hist1 = np.histogram(occurence1, bins=x_axis)
    hist8 = np.histogram(occurence8, bins=x_axis)
    print(hist1)
    plt.plot(x_axis[0:29], hist1[0])
    plt.plot(x_axis[0:29], hist8[0])
    plt.show()
    #plt.x_axis(occurence1, bins=x_axis, facecolor='green', alpha=0.5)
    #plt.x_axis(occurence8, bins=x_axis, facecolor='blue', alpha=0.5)
    
if __name__ == "__main__":
    main()