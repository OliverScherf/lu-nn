'''
Created on 10.02.2018

@author: Group 35
'''
import numpy as np

def euclidianDistance(a, b):
    return np.absolute(np.linalg.norm(a - b))

def main():
    trainIn = np.genfromtxt('data/train_in.csv', delimiter=',')
    trainOut = np.genfromtxt('data/train_out.csv', delimiter=',')
    
    meanSum = [0] * 10
    occurence =  [0] * 10
    cloud = [ [] for i in range(10) ]
    
    for i in range(0, len(trainOut)):
      actualNumber = int(trainOut[i])
      meanSum[actualNumber] += trainIn[i]
      occurence[actualNumber] += 1
      cloud[actualNumber].append(trainIn[i])
    
    center = [0] * 10
    radius = [0] * 10
    for b in range(0, 10):
      center[b] = meanSum[b] / occurence[b]
      for img in cloud[b]:
        res = euclidianDistance(center[b], img)
        if res > radius[b]:
          radius[b] = res
          
    print("radii are")
    print(radius)
    np.savetxt("radius.csv", radius, delimiter=",")
    
    
    totalDistance = np.zeros([10, 10])
    for b in range(0, 10):
      for c in range(0, 10):
        totalDistance[b][c] += euclidianDistance(center[b], center[c])
    
    print("distances are")
    print(totalDistance)
    np.savetxt("distances.csv", totalDistance, delimiter=",", fmt='%2.2f')

if __name__ == "__main__":
    main()