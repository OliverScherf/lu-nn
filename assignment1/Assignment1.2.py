'''
Created on 21.02.2018

@author: Group 35
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
from Utils import plot_confusion_matrix

def classify(img, center, distanceFunc):
  currentMin = sys.maxsize
  identifiedAs = -1
  for b in range(0, 10):
    res = distanceFunc(center[b], img)
    if currentMin > res:
      currentMin = res
      identifedAs = b
  return identifedAs


def euclidianDistance(a, b):
    return np.absolute(np.linalg.norm(a - b))

def pairDistanceMaker(algo):
    def pairDistance(a, b):
        res = sklearn.metrics.pairwise.pairwise_distances([a],[b], metric=algo)
        #print(res)
        return res
    return pairDistance

def trainWith(trainIn, trainOut, distanceFunc):
  center = [0] * 10
  meanSum = [0] * 10
  meanOccurence = [0] * 10
  cloud = [ [] for i in range(10) ]

  # fill the cloud with data and prepare fo the computation of the mean for every digit
  i = 0
  for _ in np.nditer(trainOut.T):
    actualNumber = int(trainOut[i])
    meanSum[actualNumber] += trainIn[i]
    meanOccurence[actualNumber] += 1
    cloud[actualNumber].append(trainIn[i])
    i += 1

  # calculate radius and the center for each digit
  radius = [0] * 10
  for b in range(0, 10):
    center[b] = meanSum[b] / meanOccurence[b]
    for img in cloud[b]:
      distance = distanceFunc(center[b], img)
      if distance > radius[b]:
        radius[b] = distance
  return center      
        
def classifyDataset(setIn, setOut, center, distanceFunc, fileName, newTitle, metric):
  # measure performance by creating confusion matrix
  correctClassifications = [0] * 10
  classificationAmount = [0] * 10
  confusionMatrix = np.zeros([10, 10])
  i = 0
  
  for _ in np.nditer(setOut.T):
    actualNumber = int(setOut[i])
    recognizedNumber = classify(setIn[i], center, distanceFunc)
    confusionMatrix[actualNumber][recognizedNumber] += 1
    if (actualNumber == recognizedNumber):
      correctClassifications[actualNumber] += 1
    classificationAmount[actualNumber] += 1
    i += 1
    
  sum = 0
  for i in range(0,10):
      sum += correctClassifications[i]
  
  print("correct classified with "+ metric + " " + str(sum), "Correct/Incorrect ratio:", str(sum/len(setIn)))

  # Plot non-normalized confusion matrix
  plt.figure()
  plot_confusion_matrix(confusionMatrix, classes=range(0, 10),title=newTitle)
  plt.savefig(fileName)
  
def main():
    trainIn = np.genfromtxt('data/train_in.csv', delimiter=',')
    trainOut = np.genfromtxt('data/train_out.csv', delimiter=',')
    testIn = np.genfromtxt('data/test_in.csv', delimiter=',')
    testOut = np.genfromtxt('data/test_out.csv', delimiter=',')
    
    print("Results for Training set:")
    for metric in ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']:
        distFunc = pairDistanceMaker(metric)
        center = trainWith(trainIn, trainOut, distFunc)
        classifyDataset(trainIn, trainOut, center, distFunc, "Training-Set-CM-" + metric + ".png", "Confusion matrix for training set classification (" + metric + " distance)", metric)

    print("Results for Test set:")
    for metric in ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']:
        distFunc = pairDistanceMaker(metric)
        center = trainWith(trainIn, trainOut, distFunc)
        classifyDataset(testIn, testOut, center, distFunc, "Test-Set-CM-" + metric + ".png", "Confusion matrix for test set classification (" + metric + " distance)", metric)
        
if __name__ == "__main__":
    main()