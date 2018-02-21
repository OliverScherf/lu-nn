import numpy as np
import sys

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Recognized Number')
    plt.xlabel('Actual Number')



def classify(img, center):
  currentMin = sys.maxsize
  identifiedAs = -1
  for b in range(0, 10):
    res = euclidianDistance(center[b], img)
    if currentMin > res:
      currentMin = res
      identifedAs = b
  return identifedAs


def euclidianDistance(a, b):
    return np.absolute(np.linalg.norm(a - b))

def trainWith(trainIn, trainOut, fileName, newTitle):
  center = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  meanSum = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  meanOccurence = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  cloud = [[],[],[],[],[],[],[],[],[],[]]

  # fill the cloud with data and prepare fo the computation of the mean for every digit
  i = 0
  for _ in np.nditer(trainOut.T):
    actualNumber = int(trainOut[i])
    meanSum[actualNumber] += trainIn[i]
    meanOccurence[actualNumber] += 1
    cloud[actualNumber].append(trainIn[i])
    i += 1

  # calculate radius and the center for each digit
  radius = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  for b in range(0, 10):
    center[b] = meanSum[b] / meanOccurence[b]
    for img in cloud[b]:
      distance = euclidianDistance(center[b], img)
      if distance > radius[b]:
        radius[b] = distance

  # measure performance by creating confusion matrix
  correctClassifications = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  classificationAmount = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  confusionMatrix = np.zeros([10, 10])
  i = 0
  
  for _ in np.nditer(trainOut.T):
    actualNumber = int(trainOut[i])
    recognizedNumber = classify(trainIn[i], center)
    confusionMatrix[actualNumber][recognizedNumber] += 1
    if (actualNumber == recognizedNumber):
      correctClassifications[actualNumber] += 1
    classificationAmount[actualNumber] += 1
    i += 1

  print("confusion matrix is")
  print(confusionMatrix)

  
  # Plot non-normalized confusion matrix
  plt.figure()
  plot_confusion_matrix(confusionMatrix, classes=range(0, 10),title=newTitle)
  plt.savefig(fileName)
  
def main():
    trainIn1 = np.genfromtxt('data/train_in.csv', delimiter=',')
    trainOut1 = np.genfromtxt('data/train_out.csv', delimiter=',')
    trainWith(trainIn1, trainOut1, "trainingssetconfusionmatrix.png", "Confusion matrix for training set classification")
    
    testIn = np.genfromtxt('data/test_in.csv', delimiter=',')
    testOut = np.genfromtxt('data/test_out.csv', delimiter=',')
    trainWith(testIn, testOut, "testssetconfusionmatrix.png", "Confusion matrix for test set classification")
    
if __name__ == "__main__":
    main()