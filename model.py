import os
import csv
import cv2

import sklearn
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D, Convolution2D, MaxPooling2D, Dropout, Activation
import numpy as np
import matplotlib.pyplot as plt

from preprocess import get_data, generate_batch, extract_samples_from_rows


# model
# hyperparams
cur_epoch = 0
epochs = 7
# batch_size = 100
learnrate = 1e-4 # 0.001 #

rows, cols, c = (66, 320, 3)


def model_nvidia():
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.,
              input_shape=(rows, cols, c),))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    # model.add(Dropout(.5))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    # model.add(Dropout(.5))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    # model.add(Dropout(.5))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(Activation('relu'))
    # model.add(Dropout(.5))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(Activation('relu'))
    # model.add(Dropout(.5))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Flatten())

    model.add(Dropout(.5))

    model.add(Activation('relu'))
    model.add(Dense(1164))
    model.add(Dropout(.5))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Dropout(.5))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Dropout(.5))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Dropout(.5))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.summary()

    optimizer = keras.optimizers.Adam(lr=learnrate)

    # Patch change optimizer to mae. Research
    model.compile(
        loss="mse",
        optimizer=optimizer,
    )
    return model


def model_lenet():
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.,
              input_shape=(rows, cols, c),))

    model.add(Convolution2D(6, 5, 5, border_mode="same"))
    model.add(Activation('relu'))
    # model.add(Dropout(.5))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Convolution2D(16, 5, 5, border_mode="same"))
    model.add(Activation('relu'))
    # model.add(Dropout(.5))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())

    model.add(Dropout(.5))

    model.add(Dense(120))
    model.add(Dropout(.5))
    model.add(Activation('relu'))

    model.add(Dense(84))
    model.add(Dropout(.5))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.summary()

    optimizer = keras.optimizers.Adam(lr=learnrate)

    # Patch change optimizer to mae. Research
    model.compile(
        loss="mse",
        optimizer=optimizer,
    )
    return model


def simple_model():
    model = Sequential()
    # normalize image
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    # crop image
    model.add(Cropping2D(cropping=((70, 20), (0, 0))))
    model.add(Convolution2D(6, 5, 5, border_mode='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), border_mode='valid'))
    model.add(Convolution2D(15, 5, 5, border_mode='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), border_mode='valid'))
    model.add(Dense(100))
    model.add(Dropout(0.65))
    model.add(Flatten())
    model.add(Dense(1))

    model.compile(loss='mae', optimizer='adam')

    return model


def reload_model(epoch=None):
    name = "model.h5"
    if epoch:
        name = "model.h5_{}".format(epoch)
    return keras.models.load_model(name)

