import os
import csv
import cv2

import sklearn
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D, Convolution2D, MaxPooling2D, Dropout
import numpy as np
import matplotlib.pyplot as plt

from preprocess import get_data, generate_batch

STEERING_CORRECTION_FACTORS = [0, 1.0 / 10, -1.0 / 10]

# TODO set to True
INCLUDE_SIDES = False
# set to false to also augment side images
AUGMENT_CENTER_ONLY = True

# Mulitplier for counting total training samples after augmentation
camera_sides = (3 if INCLUDE_SIDES else 1)
AUGMENTED_SAMPLE_MULTIPLIER = camera_sides + (1 if AUGMENT_CENTER_ONLY else camera_sides)

samples = []
sample_count = 0
for folder in PATHS_TO_IMG_FOLDERS:
        log_file_path = os.path.join(PATH_TO_DATA_FOLDER, folder, LOG_FILE)
        count = 0
        with open(log_file_path) as logs:
            reader = csv.reader(logs)
            for row in reader:
                if row[0] == "center":
                    continue
                row.append(folder)
                samples.append(row)
                count += 1
        if folder in MUTATE_DATA:
            count *= MUTATE_DATA[folder]
        sample_count += count

# count of total samples to be processed
sample_count *= AUGMENTED_SAMPLE_MULTIPLIER


# loading data
def get_batch(samples, side_images=True):
    """Returns next (images, angle of steering) batch"""
    while True:
        samples = sklearn.utils.shuffle(samples)
        count_b = 0
        images = []
        ang = []
        for row in samples:
            folder = row[-1]

            # fetches image from given path
            def get_img(img_entry):
                path = os.path.join(PATH_TO_DATA_FOLDER, folder, PATH_TO_IMG,
                                    img_entry.split('/')[-1])
                return cv2.imread(path)

            if side_images:
                images_to_add = [get_img(row[i]) for i in range(3)]
                angs_to_add = [float(row[3]) + offset for offset in STEERING_CORRECTION_FACTORS]
            else:
                images_to_add = [get_img(row[0]), ]
                angs_to_add = [float(row[3]), ]

            # augmenting image by flipping horizontally
            if folder in AUGMENT_DATA:
                if not AUGMENT_CENTER_ONLY:
                    # augment all cameras
                    images_to_add.extend([cv2.flip(image, 1) for image in images_to_add])
                    angs_to_add.extend([angle * -1 for angle in angs_to_add])
                else:
                    # augment center camera image only
                    images_to_add.append(cv2.flip(images_to_add[0], 1))
                    angs_to_add.append(angs_to_add[0] * -1)

            # mutating data by replicating it `n` times given by dict val
            # TODO sample weights as replacement for replication?
            if folder in MUTATE_DATA:
                images_to_add = list(map(lambda x: x.copy(),
                                         images_to_add * MUTATE_DATA[folder]))
                angs_to_add = angs_to_add * MUTATE_DATA[folder]

            # adding obtained data to list
            images.extend(images_to_add)
            ang.extend(angs_to_add)

            # yielding current batch
            if len(images) >= IMG_BATCH_SIZE:
                count_b += len(images)
                # TODO shuffle each batch?
                yield (np.array(images),
                       np.array(ang))
                images = []
                ang = []

        count_b += len(images)
        yield (np.array(images),
               np.array(ang))


# model
# hyperparams
epochs = 10
# batch_size = 100
learnrate = 0.001

# layers
layers = [
    # preprocessing
    Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3),),
    Cropping2D(cropping=((60, 25), (0, 0))),

    # conv 1
    Convolution2D(32, 5, 5, border_mode='valid', activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.5),

    # conv 2
    Convolution2D(16, 5, 5, border_mode='valid', activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.5),

    # conv 3
    Convolution2D(16, 5, 5, border_mode='valid', activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.5),

    # conv 4
    Convolution2D(8, 5, 5, border_mode='valid', activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.5),

    # conv 5
    Convolution2D(4, 5, 5, border_mode='valid', activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.5),

    # flatten
    Flatten(),

    # dense
    Dense(64, activation='relu'),
    Dropout(0.5),

    # dense
    Dense(64, activation='relu'),
    Dropout(0.5),

    # dense
    Dense(32, activation='relu'),
    Dropout(0.5),

    # dense
    Dense(1,),
]

# model compile
model = Sequential(layers)

optimizer = keras.optimizers.Adam(lr=learnrate)

model.compile(
    loss="mse",
    optimizer=optimizer,
)

# shuffling samples
# TODO change to inplace shuffling with np.random.shuffle
samples = sklearn.utils.shuffle(samples)

training_samples, validation_samples = samples[:len(samples) // 5], samples[len(samples) // 5:]

validation_sample_size = sample_count // 5
training_sample_size = sample_count - validation_sample_size

train_gen = get_batch(training_samples, INCLUDE_SIDES)
validation_gen = get_batch(validation_samples, INCLUDE_SIDES)

print ("training {} samples.".format(sample_count))

history_object = model.fit_generator(train_gen, samples_per_epoch=training_sample_size,
                                     nb_epoch=epochs, validation_data=validation_gen,
                                     nb_val_samples=validation_sample_size,)

### print the keys contained in the history object
print(history_object.history.keys())

# saving history object
import pickle
with open('history.pk', 'wb') as handle:
    pickle.dump(history_object.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

### plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()

# model.save('model.h5')

# TODO evaluation
