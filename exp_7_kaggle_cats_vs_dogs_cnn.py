# Exp - 7
# CNN applied to Kaggle Cats vs Dogs dataset,

import os
from shutil import copyfile
from shutil import rmtree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Callbacks
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=2,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
callbacks = [learning_rate_reduction]

# All files are here including train.zip and test1.zip
base_dir = './cats-v-dogs'
# This is where I am extracting train.zip, will copy from here to train/cats and
#  train/dogs and to validation/cats and validation/dogs
tmp_dir = os.path.join(base_dir, 'tmp/train')
# This is training folder
train_dir = os.path.join(base_dir, 'train')
# This is validation folder. We will copy from train 15% of the dogs to validation/dogs and 15% of cats to validation/Cats
# Validation is part of the development procss of the model, whether the sub-set of data samples are assigned
# to validation set dynamically through Corss-Validation or it is fixed set from the beginning of training till the end.
validation_dir = os.path.join(base_dir, 'validation')
# This is evaluation folder. We will copy from train 5% of the dogs to evaluation/dogs and 5% of cats to evaluation/Cats
# Evaluation data set is a sub-set of data samples that seen only after finishing the training/evaluation cycles
# over the designed Epochs. So this is never-seen data samples and it will give more accurate estimate of the model
# performance rather than relying solely on the measured performance over validation.
evaluation_dir = os.path.join(base_dir, 'evaluation')
# This is test folder, we extract test1.zip here. This is the 'Production' Dataset you can consider
# where you don't know to which class each image belongs.
test_dir = os.path.join(base_dir, 'test1')

# Directory with our training cat/dog pictures
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

CONTINUE_TRAINING = False

if not CONTINUE_TRAINING:
    # # Create folders if they are not.
    print('Creating folders ....')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    else:
        rmtree(train_dir)
        os.mkdir(train_dir)

    if not os.path.exists(train_cats_dir):
        os.mkdir(train_cats_dir)
    if not os.path.exists(train_dogs_dir):
        os.mkdir(train_dogs_dir)

    if not os.path.exists(validation_dir):
        os.mkdir(validation_dir)
    else:
        rmtree(validation_dir)
        os.mkdir(validation_dir)

    if not os.path.exists(validation_cats_dir):
        os.mkdir(validation_cats_dir)
    if not os.path.exists(validation_dogs_dir):
        os.mkdir(validation_dogs_dir)

list_of_fnames = os.listdir(tmp_dir)
list_of_cats_fnames = [i for i in list_of_fnames if 'CAT' in i.upper()]
print('Found {0} CATS images in input folder tmp/train'.format(len(list_of_cats_fnames)))
list_of_dogs_fnames = [i for i in list_of_fnames if 'DOG' in i.upper()]
print('Found {0} DOGS images in input folder tmp/train'.format(len(list_of_dogs_fnames)))

np.random.shuffle(list_of_cats_fnames)
np.random.shuffle(list_of_dogs_fnames)

TOTAL_CATS = len(list_of_cats_fnames)
TOTAL_DOGS = len(list_of_dogs_fnames)

TRAIN_VALIDATION_SPLIT_AT = 0.8

BATCH_SIZE = 100
TARGET_SIZE = (128, 128)
NO_OF_EPOCHS = 50
EXPERIMENT_SIZE = 12500  # Size of the sample set per category, cats or doags.
# This is to control how many samples we want to experiment with the model on.
# This helps to build the model incrementally by experimenting on smaller
# set size, adjust parameters and the complexity of the network, then
# to seek better performance we increase complexity of the network
# train again until we overfit, add more data, and so on untill we
# we make use of all data available.

print('\nDistributing images to \n {0} \n {1} \n {2} \n {3}'
      '\nsuch that {4}% of total number of images goes to training and \n'
      '{5}% goes to validation'.format(
    train_cats_dir, train_dogs_dir,
    validation_cats_dir, validation_dogs_dir,
    round(TRAIN_VALIDATION_SPLIT_AT * 100),
    round((1 - TRAIN_VALIDATION_SPLIT_AT) * 100)))

if not CONTINUE_TRAINING:
    c = 0
    for i in list_of_cats_fnames:
        if c < (round(TRAIN_VALIDATION_SPLIT_AT * EXPERIMENT_SIZE)):
            copyfile(os.path.join(tmp_dir, i), os.path.join(train_cats_dir, i))
        else:
            copyfile(os.path.join(tmp_dir, i), os.path.join(validation_cats_dir, i))
        c += 1
        if c >= EXPERIMENT_SIZE:
            break

    c = 0
    for i in list_of_dogs_fnames:
        if c < (round(TRAIN_VALIDATION_SPLIT_AT * EXPERIMENT_SIZE)):
            copyfile(os.path.join(tmp_dir, i), os.path.join(train_dogs_dir, i))
        else:
            copyfile(os.path.join(tmp_dir, i), os.path.join(validation_dogs_dir, i))
        c += 1
        if c >= EXPERIMENT_SIZE:
            break

print('Total training cat images :', len(os.listdir(train_cats_dir)))
print('Total training dog images :', len(os.listdir(train_dogs_dir)))

print('Total validation cat images :', len(os.listdir(validation_cats_dir)))
print('Total validation dog images :', len(os.listdir(validation_dogs_dir)))

print('Loading images through generators ...')
# # Here we create ImageDataGenerator and we normalize while loading
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)
validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1)
#
# # # We then load data through the generator
train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=TARGET_SIZE,  # Resize the image while loading
    batch_size=BATCH_SIZE,  #
    class_mode='binary')  # 1 Dimensional binary labels, generator assigns 0 to cats, and 1 to dogs
# we can see that from train_generator.model.indicies

TOTAL_TRAINING = len(train_generator.filenames)
#
validation_generator = validation_datagen.flow_from_directory(
    directory=validation_dir,  # This is the source directory for training images
    target_size=TARGET_SIZE,  # All images will be resized to 150x150
    batch_size=BATCH_SIZE,
    class_mode='binary')

TOTAL_VALIDATION = len(validation_generator.filenames)

print('Constructing and compiling model ...')
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',  # not sparse_crossentropy or categorical_corssentropy since
              # we are doing two class which can ben handled as
              # a binary classification
              metrics=['accuracy'])

if CONTINUE_TRAINING:
    # Load existing model for incremental learning if it exists
    if os.path.exists("model.h5"):
        print('loading previous model......')
        model.load_weights("model.h5")

# here we train the model
print('Training ....')
history = model.fit(
    train_generator,
    epochs=NO_OF_EPOCHS,
    validation_data=validation_generator,
    steps_per_epoch=TOTAL_TRAINING / BATCH_SIZE,
    validation_steps=TOTAL_VALIDATION / BATCH_SIZE,
    callbacks=[callbacks],
    verbose=2)  # Found that this is the clearest, no annoying progress bars


# # # Save the model
model.save_weights("model.h5")

#
# # -----------------------------------------------------------
# # Retrieve a list of list results on training and test data
# # sets for each training epoch
#
# # -----------------------------------------------------------
# To have a health training Loss should decrease while accuracy increases
# if loss increase while accuracy increases then this is an overfitting case
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
#
epochs = range(len(acc))  # Get number of epochs
#
# # # ------------------------------------------------
# # # Plot training and validation accuracy per epoch
# # # ------------------------------------------------
plt.plot(epochs, acc, color='b', label="Training accuracy")
plt.plot(epochs, val_acc, color='r', label="Validation accuracy")
plt.title('Training and validation accuracy')
plt.legend(loc='best', shadow=True)
plt.figure()

#
# ------------------------------------------------
# Plot training and validation loss per epoch
# ------------------------------------------------
plt.plot(epochs, loss, color='b', label="Training loss")
plt.plot(epochs, val_loss, color='r', label="Validation loss")
plt.title('Training and validation loss')
plt.legend(loc='best', shadow=True)
# plt.figure()
plt.savefig(os.path.join(base_dir, 'accuracy_loss.jpg'))
plt.show()
# #
#

# # Prediction for Kaggle submission,
# This is similar to when Production data is coming to the trained model to work on.
# TO be able to load all samples and and use flow_from_directory, set the classes =['test1]
# and point the folder to the parent base folder base_dir, also set the batch size = 1
# class_mode is set to binary as in training-validation, Also and very important for
# the predictions to keep matching the generatro filenames list "test_generator.filenames"
# we have to keep the shuffle flag to false

print('Production ....')
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    directory=base_dir,
    target_size=(128, 128),
    class_mode='binary',
    batch_size=1,
    classes=['test1'],
    shuffle=False)  # Very important to be set to False in order for the order of the filenames
# read by the generator to match the order of predictions generated by
# model.predict method and then we can associate predicionts with files names

NO_SMAPLES = len(test_generator.filenames)
predictions = model.predict(test_generator, steps=NO_SMAPLES)

# Preparing Kaggle submission file
predictions = np.where(predictions >= 0.5, 1, 0)  # rounding probabilities of the output
filenames = test_generator.filenames  #
Ids = [i.split('.')[0].split('\\')[1] for i in filenames]  # extract ids from filenames
labels = dict((v, k) for k, v in train_generator.class_indices.items())  # labels and their indices
predictions = [labels[k[0]] for k in predictions]
submission_df = pd.DataFrame({"id": Ids, "label": predictions})
submission_df.to_csv('submission.csv', index=False)
