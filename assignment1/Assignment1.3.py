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
    
    hist = np.linspace(0.5, 1.9, 30)
    print(hist)
    occurence1 = []
    occurence8 = []
    for i in range(0, len(filteredSetIn)):
        result = extractFeature(filteredSetIn[i])
        idx = np.where(hist == result)
        
        if (filteredSetOut[i] == 1):
            occurence1.append(result)
        else:
            occurence8.append(result)

    plt.hist(occurence1, bins=hist, facecolor='green', alpha=0.5)
    plt.hist(occurence8, bins=hist, facecolor='blue', alpha=0.5)
    print(occurence1)
    print(occurence8)

    plt.show()
    print(np.sum(occurence1))
    print(np.sum(occurence8))
    
if __name__ == "__main__":
    main()