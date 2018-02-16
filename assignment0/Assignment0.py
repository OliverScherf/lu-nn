'''
Created on 10.02.2018

@author: Oliver Scherf
'''

import numpy as np
import matplotlib.pyplot as plt

def plotFeatureSpace(xMatrix, andVector, xorVector):
    _, (ax1, ax2) = plt.subplots(1,2,figsize=(5,2.5))
    plt.setp((ax1, ax2), xticks=[0,1], yticks=(0, 1))
    ax1.plot(xMatrix[0,andVector==0],xMatrix[1,andVector==0], 'ob')
    ax1.plot(xMatrix[0,andVector==1],xMatrix[1,andVector==1], 'og')
    ax2.plot(xMatrix[0,xorVector==0],xMatrix[1,xorVector==0], 'ob')
    ax2.plot(xMatrix[0,xorVector==1],xMatrix[1,xorVector==1], 'og')

def plotDecisionBoundary(xMatrix, andVector, weights, bias):
    xAxis = np.linspace(0, 1, 1001)
    yAxis = np.linspace(0, 1, 1001)
    xx, yy = np.meshgrid(xAxis, yAxis)
    points = np.c_[xx.ravel(), yy.ravel()]
    f_hat = weights.dot(points.T) + bias
    predict = np.heaviside(f_hat, 1)
    predict = predict.T
    z = predict.reshape(xx.shape)
    plt.figure(figsize=(4,4))
    plt.plot(xMatrix[0,andVector==0], xMatrix[1,andVector==0], 'ob', xMatrix[0,andVector==1], xMatrix[1,andVector==1],'og', markersize=15)
    plt.plot()
    plt.xticks([0, 1])
    plt.yticks([0,1])
    plt.quiver(weights[0], weights[1], scale=2.0)
    plt.contour(xx,yy,z,0.1)
    plt.show()
  

def main():
    # The feature space
    xMatrix = np.array([[0, 0, 1, 1],
                        [0, 1, 0, 1]])
    
    andVector = np.array([0, 0, 0, 1])
    xorVector = np.array([0, 1, 1, 0])
    plotFeatureSpace(xMatrix, andVector, xorVector)
    
    weights = np.array([0.5, 0.5]) #use two brackets to be able to transpose
    bias = -0.5484733893874026 #np.random.randn()
    y = weights.T.dot(xMatrix) + bias
    
    result = np.heaviside(y, 0)
    print('classified: ', result)
    print('correct classified: ', result == andVector)
    plotDecisionBoundary(xMatrix, andVector, weights, bias)

if __name__ == "__main__":
    main()

