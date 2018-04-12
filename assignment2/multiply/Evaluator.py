# -*- coding: utf-8 -*-
'''An implementation of sequence to sequence learning for performing addition

Input: "535+61"
Output: "596"
Padding is handled by using a repeated sentinel character (space)

Input may optionally be reversed, shown to increase performance in many tasks in:
"Learning to Execute"
http://arxiv.org/abs/1410.4615
and
"Sequence to Sequence Learning with Neural Networks"
http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
Theoretically it introduces shorter term dependencies between source and target.

Two digits reversed:
+ One layer LSTM (128 HN), 5k training examples = 99% train/test accuracy in 55 epochs

Three digits reversed:
+ One layer LSTM (128 HN), 50k training examples = 99% train/test accuracy in 100 epochs

Four digits reversed:
+ One layer LSTM (128 HN), 400k training examples = 99% train/test accuracy in 20 epochs

Five digits reversed:
+ One layer LSTM (128 HN), 550k training examples = 99% train/test accuracy in 30 epochs
'''

from __future__ import print_function
from keras.models import Sequential
from keras import layers
from keras.models import load_model
import numpy as np
from six.moves import range
import matplotlib.pyplot as plt
import os.path


class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    """
    def __init__(self, chars):
        """Initialize character table.

        # Arguments
            chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """One hot encode given string C.

        # Arguments
            num_rows: Number of rows in the returned one hot encoding. This is
                used to keep the # of rows for each data the same.
        """
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

# Parameters for the model and dataset.
TRAINING_SIZE = 200
DIGITS = 3
REVERSE = False

# Maximum length of input is 'int + int' (e.g., '345+678'). Maximum length of
# int is DIGITS.
MAXLEN = DIGITS + 1 + DIGITS

# All the numbers, plus sign and space for padding.
chars = '0123456789* '
ctable = CharacterTable(chars)
ANSWER_LENGTH = 6
questions = []
expected = []
seen = set()
print('Generating data...')
while len(questions) < TRAINING_SIZE:
    f = lambda: int(''.join(np.random.choice(list('0123456789'))
                    for i in range(np.random.randint(1, DIGITS + 1))))
    a, b = f(), f()
    # Skip any addition questions we've already seen
    # Also skip any such that x+Y == Y+x (hence the sorting).
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    # Pad the data with spaces such that it is always MAXLEN.
    q = '{}*{}'.format(a, b)
    query = q + ' ' * (MAXLEN - len(q))
    ans = str(a * b)
    while (len(ans) < 6):
      ans = "0" + ans
    # Answers can be of maximum size DIGITS + 1.
    ans += ' ' * (DIGITS + 1 - len(ans))
    if REVERSE:
        # Reverse the query, e.g., '12+345  ' becomes '  543+21'. (Note the
        # space used for padding.)
        query = query[::-1]
    questions.append(query)
    expected.append(ans)
print('Total addition questions:', len(questions))

print('Vectorization...')
x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(questions), ANSWER_LENGTH, len(chars)), dtype=np.bool)
for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, MAXLEN)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, ANSWER_LENGTH)

# Shuffle (x, y) in unison as the later parts of x will almost all be larger
# digits.
indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# Explicitly set apart 10% for validation data that we never train over.
split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]

print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)

# Try replacing GRU, or SimpleRNN.
RNN = layers.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1

def getSampleData(sampleSize):
    toTry = []
    for i in range(sampleSize):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
        toTry.append((rowx, rowy))
    return toTry


def evaluateModel(modelName, sampleSize, isStatic=False, toTry=[]):
    accuracies = []
    if len(toTry) == 0:
        toTry = getSampleData(sampleSize)
    iteration = 1
    while os.path.isfile(modelName + "_" + str(iteration) + ".h5") or (isStatic and iteration == 1):
        model = load_model(modelName if isStatic else (modelName + "_" + str(iteration) + ".h5"))
        correctResult = 0
        for i in range(len(toTry)):
            rowx = toTry[i][0]
            rowy = toTry[i][1]
            preds = model.predict_classes(rowx, verbose=0)
            q = ctable.decode(rowx[0])
            correct = ctable.decode(rowy[0])
            guess = ctable.decode(preds[0], calc_argmax=False)
            if correct == guess:
                correctResult += 1
        accuracies.append(correctResult / len(toTry))
        iteration += 1
    return accuracies


def plotAccuracies(fileName, accuracies, xTicks, xTicks2=None):
    plt.figure()
    for accuracy in accuracies:
        plt.plot(accuracy[1], label=accuracy[0])
    plt.xticks(xTicks, xTicks2)
    plt.yticks(np.arange(0, 1.0, 0.05))
    plt.title("Accuracies")
    tick_marks = np.arange(len(accuracies))
    plt.ylabel('Accuracy')
    plt.xlabel('Iteration')
    plt.savefig(fileName)

sample = getSampleData(1000)
accuraciesOfModels = [("GRU 1", evaluateModel("gru1/gru1", 1, False, sample))]
plotAccuracies("accuarcy_by_iteration,png", accuraciesOfModels, np.arange(len(20)))

modelNames = ["mult-0.h5"]
sampleSizesToTest = [3, 5, 10]
sampleLabels = map(str, sampleSizesToTest)
samplesToTest = []
for sampleSize in sampleSizesToTest:
    samplesToTest.append(getSampleData(sampleSize))

samplePlotData = []
for modelName in modelNames:
    results = []
    for sampleData in samplesToTest:
        results.append(evaluateModel(modelName, 1, 1, True, sampleData)[0])
        print("results now is", results)
    samplePlotData.append((modelName, results))
    print("sampleplotdata is", samplePlotData)


plotAccuracies("accuarcy_by_samplesize,png",samplePlotData, np.arange(len(samplesToTest)), sampleLabels)
               
               
               
               