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

STEERING_CORRECTION_FACTORS = [0, 1.0 / 10, -1.0 / 10]

# TODO set to True
INCLUDE_SIDES = False
# set to false to also augment side images
AUGMENT_CENTER_ONLY = True

# Mulitplier for counting total training samples after augmentation
camera_sides = (3 if INCLUDE_SIDES else 1)

# model
# hyperparams
epochs = 7
# batch_size = 100
learnrate = 0.001

def create_new_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.,
              input_shape=(34, 160, 3),))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Dropout(.5))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Dropout(.5))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Dropout(.5))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(Activation('relu'))
    model.add(Dropout(.5))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(Activation('relu'))
    model.add(Dropout(.5))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Flatten())
    model.add(Dropout(.2))
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


def reload_model():
    return keras.models.load_model("model.h5")


model = create_new_model()

# shuffling samples
# TODO change to inplace shuffling with np.random.shuffle
samples, sample_count = get_data()

validation_samples, training_samples = samples[:len(samples) // 100], samples[len(samples) // 100:]

validation_sample_size = sample_count // 100
# TODO add modifier
training_sample_size = sample_count - (validation_sample_size)

# , INCLUDE_SIDES
train_gen = generate_batch(training_samples, augment=False, flipit=True, rand_camera=True)
validation_gen = generate_batch(validation_samples, False, flipit=True, rand_camera=True)

print ("training {} samples.".format(sample_count))

history_object = model.fit_generator(train_gen, samples_per_epoch=training_sample_size,
                                     nb_epoch=epochs, validation_data=validation_gen,
                                     nb_val_samples=validation_sample_size,)

print("training finished!!!!!!!!!!!!")
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save("model.h5")

# saving history object
import pickle
with open('history.pk', 'wb') as handle:
    pickle.dump(history_object.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
