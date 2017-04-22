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
cur_epoch = 0
epochs = 7
# batch_size = 100
learnrate = 1e-4 # 0.001 #

rows, cols, c = (66, 320, 3)

def create_new_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.,
              input_shape=(rows, cols, c),))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    #model.add(Dropout(.5))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    #model.add(Dropout(.5))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    #model.add(Dropout(.5))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(Activation('relu'))
    #model.add(Dropout(.5))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(Activation('relu'))
    #model.add(Dropout(.5))
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

def lenet():
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.,
              input_shape=(rows, cols, c),))

    model.add(Convolution2D(6, 5, 5, border_mode="same"))
    model.add(Activation('relu'))
    #model.add(Dropout(.5))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Convolution2D(16, 5, 5, border_mode="same"))
    model.add(Activation('relu'))
    #model.add(Dropout(.5))
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

    #model.summary()

    optimizer = keras.optimizers.Adam(lr=learnrate)

    # Patch change optimizer to mae. Research
    model.compile(
        loss="mse",
        optimizer=optimizer,
    )
    return model



def reload_model(epoch=None):
    name = "model.h5"
    if epoch:
        name = "model.h5_{}".format(epoch)
    return keras.models.load_model(name)


model = create_new_model()
# model = reload_model(cur_epoch)

# shuffling samples
# TODO change to inplace shuffling with np.random.shuffle
#samples, sample_count = get_data()

training_samples, training_sample_size, validation_samples, validation_sample_size = get_data(split_valid=0.2)

#validation_samples, training_samples = samples[:len(samples) // 100], samples[len(samples) // 100:]

#validation_sample_size = sample_count // 100
# TODO add modifier
#training_sample_size = sample_count - (validation_sample_size)

# , INCLUDE_SIDES
train_gen = generate_batch(training_samples)
validation_gen = generate_batch(validation_samples, augment=False, flipit=True)

print ("training {} samples.".format(training_sample_size))
print ("validating {} samples.".format(validation_sample_size))

for i in range(cur_epoch, epochs):

    history_object = model.fit_generator(train_gen, samples_per_epoch=training_sample_size,
                                         nb_epoch=1, validation_data=validation_gen,
                                         nb_val_samples=validation_sample_size,)
    model_json = model.to_json()
    # with open("model.json", "w") as json_file:
    #     json_file.write(model_json)
    print("saving model state at epoch: {}".format(i+1))
    model.save("model.h5_{}".format(i+1))

print("training finished!!!!!!!!!!!!")
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save("model.h5")

# saving history object
import pickle
with open('history.pk', 'wb') as handle:
    pickle.dump(history_object.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
