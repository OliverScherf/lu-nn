from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist

import matplotlib.pyplot as plt
import Utils
import numpy as np
from reportlab.graphics.charts.axes import _findMax



def main():
    mpl_categorial_cross_non_perm = np.genfromtxt('results/mlp_categorical_crossentropy_non_permuted.csv', delimiter=',')
    mpl_categorial_cross_perm = np.genfromtxt('results/mlp_categorical_crossentropy_permuted.csv', delimiter=',')
    mpl_mean_non_perm = np.genfromtxt('results/mlp_mean_squared_error_non_permuted.csv', delimiter=',')
    mpl_mean_perm = np.genfromtxt('results/mlp_mean_squared_error_permuted.csv', delimiter=',')
    
    cnn_categorial_cross_non_perm = np.genfromtxt('results/cnn_categorical_crossentropy_non_permuted.csv', delimiter=',')
    cnn_categorial_cross_perm = np.genfromtxt('results/cnn_categorical_crossentropy_permuted.csv', delimiter=',')
    cnn_mean_non_perm = np.genfromtxt('results/cnn_mean_squared_error_non_permuted.csv', delimiter=',')
    cnn_mean_perm = np.genfromtxt('results/cnn_mean_squared_error_permuted.csv', delimiter=',')
    
    (_, _), (x_test, y_test) = mnist.load_data()
    
    
    print("mpl_categorial_cross_non_perm")
    evaluate(y_test, mpl_categorial_cross_non_perm)
    findNWorstImages(x_test, y_test, mpl_categorial_cross_non_perm, 3, "mpl_categorial_cross_non_perm")

    print("mpl_categorial_cross_perm")
    evaluate(y_test, mpl_categorial_cross_perm)
    findNWorstImages(x_test, y_test, mpl_categorial_cross_perm, 3, "mpl_categorial_cross_perm")
    
    print("mpl_mean_non_perm")
    evaluate(y_test, mpl_mean_non_perm)
    findNWorstImages(x_test, y_test, mpl_mean_non_perm, 3, "mpl_mean_non_perm")
    
    print("mpl_mean_perm")
    evaluate(y_test, mpl_mean_perm)
    findNWorstImages(x_test, y_test, mpl_mean_perm, 3, "mpl_mean_perm")
    print("\n")
    
    print("cnn_categorial_cross_non_perm")
    evaluate(y_test, cnn_categorial_cross_non_perm)
    findNWorstImages(x_test, y_test, cnn_categorial_cross_non_perm, 3, "cnn_categorial_cross_non_perm")
    
    print("cnn_categorial_cross_perm")
    evaluate(y_test, cnn_categorial_cross_perm)
    findNWorstImages(x_test, y_test, cnn_categorial_cross_perm, 3, "cnn_categorial_cross_perm")
    
    print("cnn_mean_non_perm")
    evaluate(y_test, cnn_mean_non_perm)
    findNWorstImages(x_test, y_test, cnn_mean_non_perm, 3, "cnn_mean_non_perm")
    
    print("cnn_mean_perm")
    evaluate(y_test, cnn_mean_perm)
    findNWorstImages(x_test, y_test, cnn_mean_perm, 3, "cnn_mean_perm")
    
def evaluate(y_test, results):
    misclassified = np.zeros(10)
    for i in range(0, len(y_test)):
        recognizedAs = Utils.maxIndex(results[i])
        if (recognizedAs != y_test[i]):
            misclassified[y_test[i]] += 1
    print("Total misclassified:", np.sum(misclassified)) 
    print(misclassified)
    
    percentageOff = np.zeros(10)
    samplesPerDigit = np.zeros(10)
    
    for i in range(0, len(y_test)):
        samplesPerDigit[y_test[i]] += 1
        percentageOff[y_test[i]] += results[i][y_test[i]]
        
    percentage = percentageOff/samplesPerDigit * 100 
    print(np.round(percentage, 2))
    
def findNWorstImages(x_test, y_test, results, n, fileName):
    w, h = 2, len(y_test);
    probabilties = [[0 for x in range(w)] for y in range(h)] 
    #probabilties = np.zeros((len(y_test), 2))
    for i in range(0, len(y_test)):
        idx = Utils.maxIndex(results[i])
        probabilties[i][0] = i
        probabilties[i][1] = results[i][idx]
    
    sortedPro = sorted(probabilties,key=lambda l:l[1])

    plt.figure()
    for i in range(n):
    # display original
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(x_test[sortedPro[i][0]].reshape(28, 28))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig("digits/" + fileName + ".png")
    
def plotImg(img):
    plt.imshow(img.reshape(28, 28))

       
    
if __name__ == "__main__":
    main()
