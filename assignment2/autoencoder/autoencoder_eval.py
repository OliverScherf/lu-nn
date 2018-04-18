from keras.datasets import mnist
from keras import regularizers
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.models import load_model

import matplotlib.pyplot as plt
import numpy as np
import os.path




def autoencoder_conv():
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

    autoencoder = load_model("results/conv.h5")
    
    decoded_imgs = autoencoder.predict(x_test)
    calcDistances(x_test, decoded_imgs, "conv")
    n = 11 +  + startIndex
    plt.figure(figsize=(20, 4))
    for i in range(1 + startIndex, n):
        # display original
        ax = plt.subplot(3, n, i)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        # display reconstruction
        ax = plt.subplot(3, n, i + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display heatmap
        ax = plt.subplot(3, n, i + n * 2)
        toShow = x_test[i] - decoded_imgs[i]
        for j in range(len(toShow)):
            for k in range(len(toShow[j])):
                print(toShow[j][k])
                if (toShow[j][k] < 0.05):
                    toShow[j][k] = 1.0
        plt.imshow(toShow.reshape(28, 28))
        print(np.max(x_test[i]))
        print(np.max(decoded_imgs[i]))
        #print(np.min(x_test[i] - decoded_imgs[i]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig("autoencoder_conv.png")
    
def autoencoder_conv_mod():
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

    autoencoder = load_model("results/conv_mod.h5")
    decoded_imgs = autoencoder.predict(x_test)
    calcDistances(x_test, decoded_imgs, "conv_mod")
    n = 11  + startIndex
    plt.figure(figsize=(20, 4))
    for i in range(1 + startIndex, n):
        # display original
        ax = plt.subplot(3, n, i)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        # display reconstruction
        ax = plt.subplot(3, n, i + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display heatmap
        ax = plt.subplot(3, n, i + n * 2)
        toShow = x_test[i] - decoded_imgs[i]
        for j in range(len(toShow)):
            for k in range(len(toShow[j])):
                print(toShow[j][k])
                if (toShow[j][k] < 0.05):
                    toShow[j][k] = 1.0
        plt.imshow(toShow.reshape(28, 28))
        print(np.max(x_test[i]))
        print(np.max(decoded_imgs[i]))
        #print(np.min(x_test[i] - decoded_imgs[i]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig("autoencoder_conv_mod.png")

def autoencoder_noise():
    (x_train, _), (x_test, _) = mnist.load_data()
    
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

    noise_factor = 0.5
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
    
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)
    autoencoder = load_model("results/conv_noise.h5")
    decoded_imgs = autoencoder.predict(x_test)
    calcDistances(x_test, decoded_imgs, "noise")
    n = 11  + startIndex
    plt.figure(figsize=(20, 4))
    for i in range(1 + startIndex, n):
        # display original
        ax = plt.subplot(3, n, i)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display noisy
        ax = plt.subplot(3, n, i + n)
        plt.imshow(x_test_noisy[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        # display reconstruction
        ax = plt.subplot(3, n, i + n * 2)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig("autoencoder_noise.png")

def autoencoder_noise_mod():
    (x_train, _), (x_test, _) = mnist.load_data()
    
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

    noise_factor = 0.5
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
    
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)
    
    autoencoder = load_model("results/conv_noise_mod.h5")
    decoded_imgs = autoencoder.predict(x_test)
    calcDistances(x_test, decoded_imgs, "noise_mod")
    
    n = 11  + startIndex
    plt.figure(figsize=(20, 4))
    for i in range(1 + startIndex, n):
        # display original
        ax = plt.subplot(3, n, i)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display noisy
        ax = plt.subplot(3, n, i + n)
        plt.imshow(x_test_noisy[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        # display reconstruction
        ax = plt.subplot(3, n, i + n * 2)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig("autoencoder_noise_mod.png")

def autoencoder_deep():
    (x_train, _), (x_test, _) = mnist.load_data()
    
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    encoder = load_model("results/encoder_deep.h5")
    decoder = load_model("results/decoder_deep.h5")
    # encode and decode some digits
    # note that we take them from the *test* set
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)
    calcDistances(x_test, decoded_imgs, "deep")
    
    n = 11  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(1 + startIndex, n):
        # display original
        ax = plt.subplot(3, n, i)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        # display reconstruction
        ax = plt.subplot(3, n, i + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display heatmap
        ax = plt.subplot(3, n, i + n * 2)
        toShow = x_test[i] - decoded_imgs[i]
        for j in range(len(toShow)):
            print(toShow[j])
            if (toShow[j] < 0.05):
                toShow[j] = 1.0
        plt.imshow(toShow.reshape(28, 28))
        print(np.max(x_test[i]))
        print(np.max(decoded_imgs[i]))
        #print(np.min(x_test[i] - decoded_imgs[i]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


    plt.savefig("autoencoder_deep.png")
    
def autoencoder_deep_mod():
    (x_train, _), (x_test, _) = mnist.load_data()
    
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    encoder = load_model("results/encoder_deep_mod.h5")
    decoder = load_model("results/decoder_deep_mod.h5")
    # encode and decode some digits
    # note that we take them from the *test* set
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)
    calcDistances(x_test, decoded_imgs, "deep_mod")
    
    n = 11  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(1 + startIndex, n):
        # display original
        ax = plt.subplot(3, n, i)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        # display reconstruction
        ax = plt.subplot(3, n, i + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display heatmap
        ax = plt.subplot(3, n, i + n * 2)
        toShow = x_test[i] - decoded_imgs[i]
        for j in range(len(toShow)):
            print(toShow[j])
            if (toShow[j] < 0.05):
                toShow[j] = 1.0
        plt.imshow(toShow.reshape(28, 28))
        print(np.max(x_test[i]))
        print(np.max(decoded_imgs[i]))
        #print(np.min(x_test[i] - decoded_imgs[i]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


    plt.savefig("autoencoder_deep_mod.png")
    
def calcDistances(orig, decoded, fileName):
    distances = []
    for i in range(len(orig)):
        distances.append(np.absolute(np.linalg.norm(orig[i] - decoded[i])))
    
    np.savetxt("results/" + fileName + ".csv", distances, delimiter=",")
    
def printDistances():
    deep = np.mean(np.loadtxt("results/deep.csv", delimiter=','))
    deep_mod = np.mean(np.loadtxt("results/deep_mod.csv", delimiter=','))
    noise = np.mean(np.loadtxt("results/noise.csv", delimiter=','))
    noise_mod = np.mean(np.loadtxt("results/noise_mod.csv", delimiter=','))
    conv = np.mean(np.loadtxt("results/conv.csv", delimiter=','))
    conv_mod = np.mean(np.loadtxt("results/conv_mod.csv", delimiter=','))
    means = [deep, deep_mod, noise, noise_mod, conv, conv_mod]
    
    x = ['deep', 'deep_mod', 'noise', 'noise_mod', 'conv', 'conv_mod']
    means = [5, 6, 15, 22, 24, 8]
    
    x_pos = [i for i, _ in enumerate(x)]
    
    plt.bar(x_pos, means, color='green')
    plt.xlabel("Energy Source")
    plt.ylabel("Energy Output (GJ)")
    plt.title("Energy output from various fuel sources")
    
    plt.xticks(x_pos, x)
    plt.savefig("distances.png")
    

startIndex = 0

autoencoder_conv()
autoencoder_conv_mod()
autoencoder_noise()
autoencoder_noise_mod()
autoencoder_deep()
autoencoder_deep_mod()
printDistances()
