'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
import numpy as np
import sys
import Utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 128
num_classes = 10
epochs = 20
seed = 1337

def getMNISTData():
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    return (x_train, y_train, x_test, y_test)
    

def trainModel(x_train, y_train, x_test, y_test, lossFunction):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.summary()
    
    model.compile(loss=lossFunction,
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    return (model, x_test, y_test)
    
def main():
    (x_train, y_train, x_test, y_test) = getMNISTData()
    (model, x_test, y_test) = trainModel(x_train, y_train, x_test, y_test, 'categorical_crossentropy')
    Utils.findMostMisclassifiedDigits(model, x_test, y_test, "mlp_categorical_crossentropy_non_permuted")

    (x_train, y_train, x_test, y_test) = getMNISTData()
    (model, x_test, y_test) = trainModel(x_train, y_train, x_test, y_test, 'mean_squared_error')
    Utils.findMostMisclassifiedDigits(model, x_test, y_test, "mlp_mean_squared_error_non_permuted")

    (x_train, y_train, x_test, y_test) = getMNISTData()
    x_train = Utils.permutateData(x_train, seed)
    x_test = Utils.permutateData(x_test, seed)
    (model, x_test, y_test) = trainModel(x_train, y_train, x_test, y_test, 'categorical_crossentropy')
    Utils.findMostMisclassifiedDigits(model, x_test, y_test, "mlp_categorical_crossentropy_permuted")
    
    (x_train, y_train, x_test, y_test) = getMNISTData()
    x_train = Utils.permutateData(x_train, seed)
    x_test = Utils.permutateData(x_test, seed)
    (model, x_test, y_test) = trainModel(x_train, y_train, x_test, y_test, 'mean_squared_error')
    Utils.findMostMisclassifiedDigits(model, x_test, y_test, "mlp_mean_squared_error_permuted")
    
    
    
 
 
if __name__ == "__main__":
    main()
