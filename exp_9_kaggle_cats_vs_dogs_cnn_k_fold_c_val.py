# Exp - 9
# DNN applied to Kaggle Cats vs Dogs dataset, images are read using ImageDataGenerator,
# and I manipulated the data set so that Cross Validation is used in training.
# Image augmentation is applied as well.
# The final NN architecture I concluded Exp 7
# [TODO][Add link to data set here]

import os
from shutil import copyfile
from shutil import rmtree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Callbacks
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
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

# This is training folder.
# We will copy 90% of dogs images from input tmp/train folder to train/dogs and
# 90% of cats to train/Cats. Cross validation is part of the development process
# of the model and sub-set of data samples are assigned validation set dynamically
# through Cross-Validation

train_dir = os.path.join(base_dir, 'train')

# This is the test folder. We will copy from input tmp/train folder 10% of the dogs
# to test/dogs and 10% of cats to test/Cats

test_dir = os.path.join(base_dir, 'test')

# This is kaggle test folder, we extract test1.zip here. This is the 'Production'
# Dataset you can consider where you don't know to which class each image belongs.
kaggle_test_dir = os.path.join(base_dir, 'test1')

# Directory with our training cat/dog pictures
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')

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

if not os.path.exists(test_dir):
    os.mkdir(test_dir)
else:
    rmtree(test_dir)
    os.mkdir(test_dir)

if not os.path.exists(test_cats_dir):
    os.mkdir(test_cats_dir)
if not os.path.exists(test_dogs_dir):
    os.mkdir(test_dogs_dir)

list_of_fnames = os.listdir(tmp_dir)
list_of_cats_fnames = [i for i in list_of_fnames if 'CAT' in i.upper()]
print('Found {0} CATS images in input folder tmp/train'.format(len(list_of_cats_fnames)))
list_of_dogs_fnames = [i for i in list_of_fnames if 'DOG' in i.upper()]
print('Found {0} DOGS images in input folder tmp/train'.format(len(list_of_dogs_fnames)))

np.random.shuffle(list_of_cats_fnames)
np.random.shuffle(list_of_dogs_fnames)

TOTAL_CATS = len(list_of_cats_fnames)
TOTAL_DOGS = len(list_of_dogs_fnames)

K_FOLDS = 5
TRAINING_TEST_SPLIT_AT = 0.9
BATCH_SIZE = 5
TARGET_SIZE = (128, 128)
NO_OF_EPOCHS = 5
EXPERIMENT_SIZE = 1000  # Size of the sample set per category, cats or doags.
# This is to control how many samples we want to experiment with the model on.
# This helps to build the model incrementally by experimenting on smaller
# set size, adjust parameters and the complexity of the network, then
# to seek better performance we increase complexity of the network
# train again until we overfit, add more data, and so on untill we
# we make use of all data available.

print('\nDistributing images to \n {0} \n {1} \n {2} \n {3} \n'
      'such that 90% of total number of images goes to training and \n'
      '10% goes to test, training is later distributed dynamically at each '
      'epoch 80-20 for training and validation'.format(
    train_cats_dir, train_dogs_dir,
    test_cats_dir, test_dogs_dir))

c = 0
for i in list_of_cats_fnames:
    if c < (round(TRAINING_TEST_SPLIT_AT * EXPERIMENT_SIZE)):
        copyfile(os.path.join(tmp_dir, i), os.path.join(train_cats_dir, i))
    else:
        copyfile(os.path.join(tmp_dir, i), os.path.join(test_cats_dir, i))
    c += 1
    if c >= EXPERIMENT_SIZE:
        break

c = 0
for i in list_of_dogs_fnames:
    if c < (round(TRAINING_TEST_SPLIT_AT * EXPERIMENT_SIZE)):
        copyfile(os.path.join(tmp_dir, i), os.path.join(train_dogs_dir, i))
    else:
        copyfile(os.path.join(tmp_dir, i), os.path.join(test_dogs_dir, i))
    c += 1
    if c >= EXPERIMENT_SIZE:
        break

print('Total training cat images :', len(os.listdir(train_cats_dir)))
print('Total training dog images :', len(os.listdir(train_dogs_dir)))

print('Total test cat images :', len(os.listdir(test_cats_dir)))
print('Total test dog images :', len(os.listdir(test_dogs_dir)))

print('Loading images through generators ...')

# # Here we create ImageDataGenerator and we normalize while loading
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
#
# # # We then load data through the generator
train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=TARGET_SIZE,  # Resize the image while loading
    batch_size=BATCH_SIZE,  #
    class_mode='binary',
    shuffle=False)  # 1 Dimensional binary labels, generator assigns 0 to cats, and 1 to dogs
# we can see that from train_generator.model.indicies

TOTAL_TRAINING = len(train_generator.filenames)

#
test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=TARGET_SIZE,
    batch_size=1,  # one sample at a time for evaluation
    class_mode='binary'
)
#
TOTAL_TEST = len(test_generator.filenames)

print('Constructing and compiling model ...')
# TODO use NN archicture concluded from Exp - 7
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2, 2),
    # tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2, 2),
    # tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2, 2),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    # tf.keras.layers.Dense(1024, activation='relu'),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',  # not sparse_crossentropy or categorical_corssentropy since
              # we are doing two class which can ben handled as
              # a binary classification
              metrics=['accuracy'])

# Load existing model for incremental learning if it exists
# if os.path.exists("model.h5"):
#     print('loading previous model......')
#     model.load_weights("model.h5")

# here we train the model
print('Training ....')

# TODO fill in a dataframe with filenames and classifications to either (0 'Cats', 1 'Dogs)
# TODO split the datafame randomely at each epoch iteration to validation and training
# TODO supply validation and training data frame to model.fit method and set epochs = 1


kf = KFold(n_splits=K_FOLDS, random_state=None, shuffle=True)
kf.get_n_splits(train_generator.filenames)

X = np.array(train_generator.filenames)
labels = dict((k, v) for k, v in train_generator.class_indices.items())
Y = np.array([labels[os.path.dirname(train_generator.filenames[i])]
              for i in (0, len(train_generator.filenames))])
# Labels will be redefined again at the end of the scritp to use keys in place of values
# for the submission

# df["category"] = df["category"].replace({0: 'cat', 1: 'dog'})

for i in range(NO_OF_EPOCHS):
    for train_index, test_index in kf.split(X):
        trainData = X[train_index]
        testData = X[test_index]
        trainLabels = Y[train_index]
        testLabels = Y[test_index]
        print("=========================================")
        print("====== K Fold Validation step => %d/%d =======" % (i,k_folds))
        print("=========================================")

        df = pd.DataFrame({
            'filename': trainData,
            'category': trainLabels
        })

        trainGenerator = Generator(trainData,trainLabels,
                                   batchSize=batchSize,imageSize=imageSize,
                                   augment=True,
                                   grayMode=grayMode)
        valGenerator = Generator(testData,testLabels,
                                 batchSize=batchSize,imageSize=imageSize,
                                 augment=False,
                                 grayMode=grayMode)


    history = model.fit(
        train_generator,
        epochs=1,
        validation_data=validation_generator,
        steps_per_epoch=TOTAL_TRAINING / BATCH_SIZE,
        validation_steps=TOTAL_VALIDATION / BATCH_SIZE,
        verbose=2)  # Found that this is the clearest, no annoying progress bars
#
# # -----------------------------------------------------------
# # Retrieve a list of list results on training and test data
# # sets for each training epoch
#
# # -----------------------------------------------------------
# To have a health training Loss should dcrease while accuracy increases
# if loss increase while accuracy increases then this is an overfitting case
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
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
# plt.figure()
# #
# # # ------------------------------------------------
# # # Plot training and validation loss per epoch
# # # ------------------------------------------------
# # plt.plot(epochs, loss, color='b', label="Training loss")
# # plt.plot(epochs, val_loss, color='r', label="Validation loss")
# # plt.title('Training and validation loss')
#
# # plt.legend(loc='best', shadow=True)
plt.show()
# #
# # # Save the model
model.save_weights("model.h5")
#
# Test on a sample of never seen images
print('Testing ....')
# TODO change this to avoid looping
testing_results = model.evaluate(test_generator)
for i in range(len(model.metrics_names)):
    if 'accuracy' in model.metrics_names[i]:
        print('Accuracy is {0}'.format(testing_results[i]))

# # Prediction for Kaggle submission,
# This is similar to when Production data is coming to the trained model to work on.

print('Production ....')
kaggle_test_datagen = ImageDataGenerator(rescale=1.0 / 255)
kaggle_test_generator = kaggle_test_datagen.flow_from_directory(
    directory=base_dir,
    target_size=(128, 128),
    class_mode='binary',
    batch_size=1,
    classes=['test1'],
    shuffle=False)  # Very important to be set to False in order for the order of the filenames
# read by the generator to match the order of predictions generated by
# model.predict method and then we can associate predicionts with files names

NO_SMAPLES = len(kaggle_test_generator.filenames)
predictions = model.predict(kaggle_test_generator, steps=NO_SMAPLES)

# Preparing Kaggle submission file
predictions = np.where(predictions >= 0.5, 1, 0)  # rounding probabilities of the output
filenames = kaggle_test_generator.filenames  #
Ids = [i.split('.')[0].split('\\')[1] for i in filenames]  # extract ids from filenames
labels = dict((v, k) for k, v in train_generator.class_indices.items())  # labels and their indices
predictions = [labels[k[0]] for k in predictions]
submission_df = pd.DataFrame({"id": Ids, "label": predictions})
submission_df.to_csv('submission.csv', index=False)
