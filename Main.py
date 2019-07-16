import NeuralNetwork
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from PIL import Image

import tensorflow as tf


def load_dataset(path, random_shuffle=True, rgb2gray=False, invert_gray=False):
    x = []
    y = []
    print("Loading data from: " + path)
    for num in os.listdir(path):
        for image in os.listdir(path + num):
            im = Image.open(path + num + '/' + image)
            im = np.asarray(im)
            if rgb2gray:
                im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
                im = cv2.bitwise_not(im)
            if invert_gray:
                im = 255 - im
            im = (im / 127.5) - 1
            im = im.reshape((28 * 28, 1))
            x.append(im)
            y.append(np.array([0 if int(num) != y_targ else 1 for y_targ in range(10)]).reshape(10, 1))
        print('Loaded: ' + num)
    print()

    if random_shuffle:
        shuffled = list(zip(x, y))
        random.shuffle(shuffled)
        split = lambda z: ([curr[0] for curr in z], [curr[1] for curr in z])
        x, y = split(shuffled)

    return np.array(x), np.array(y)


def target_nums_to_ndarray(target):
    return np.array([
        [1 if target[sample] == num else 0 for num in range(10)] for sample in range(target.shape[0])]
    ).reshape(target.shape[0], 10, 1)


def create_model():
    NN = NeuralNetwork.NeuralNetwork()
    NN.add_input_layer(28 * 28)
    NN.add_hidden_layer(64)
    # NN.add_hidden_layer(64)
    # NN.add_hidden_layer(64)
    NN.add_hidden_layer(10, act_func='softmax')
    return NN


def addPoint(loss, show=True):
    average_loss.append(loss)
    positions.append(len(average_loss))
    if show:
        plt.plot(positions, average_loss)
        plt.show()
        plt.pause(0.0001)


def feed_data(NN, x_train, y_train, x_val=None, y_val=None, train=True, num_epochs=3, batch_size=256):
    if train:
        for epoch in range(num_epochs):
            print('Epoch ' + str(epoch + 1) + '/' + str(num_epochs) + ':')
            loss, correct = NN.fit(x_train, y_train, x_val=x_val, y_val=y_val, batch_size=batch_size)
            addPoint(loss)
    else:
        predictions = NN.predict(x_train)
        loss = NN.calculate_loss(predictions, y_train)
        addPoint(loss)


def view_false(NN, x, y):
    predictions = NN.predict(x)
    for num in range(predictions.shape[0]):
        if predictions[num:num + 1, :, :].argmax() != y[num:num + 1, :].argmax():
            print(predictions[num:num + 1, :, :].argmax(), y[num:num + 1, :].argmax())
            im = (x[num:num + 1, :, :].reshape(28, 28) + 1) * 127.5
            name = "pred/target : " + str(predictions[num:num + 1, :, :].argmax()) + \
                   "/" + str(y[num:num + 1, :].argmax())
            cv2.imshow(name, im)
            cv2.waitKey()
            cv2.destroyAllWindows()


def prediction_matrix(NN, x, y):
    mat = np.zeros(shape=(10, 10))
    for num in range(x.shape[0]):
        pred = NN.predict(x[num:num + 1, :, :])
        mat[y[num:num + 1, :].argmax(), pred.argmax()] += 1
    attributes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    df = pd.DataFrame(np.around(mat / mat.sum(axis=1, keepdims=True), decimals=3), columns=attributes, index=attributes)
    print('\nTarget\\Predicted')
    print(df)

    return mat


# Directories
im_path = 'Images/'
models_path = 'Weights/'

# Setting up the plot
average_loss = []  # y-coordinates
positions = []  # x-coordinates
plt.ion()
figure = plt.figure()

NN = create_model()

# Loading-, reshaping- and normalizing the data
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
x_train = (x_train.reshape(x_train.shape[0], 28 * 28, 1) / 127.5) - 1
x_val = (x_val.reshape(x_val.shape[0], 28 * 28, 1) / 127.5) - 1
y_train = target_nums_to_ndarray(y_train)
y_val = target_nums_to_ndarray(y_val)


# NN.load_weights('trained-' + str(len(os.listdir(models_path))))
feed_data(NN, x_train, y_train, x_val=x_val, y_val=y_val, train=True, num_epochs=10, batch_size=32)
prediction_matrix(NN, x_val, y_val)
NN.save_weights('trained-' + str(len(os.listdir(models_path)) + 1))
view_false(NN, x_val, y_val)
