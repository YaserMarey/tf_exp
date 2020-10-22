# Exp-3
# Callback function allows us to stop the training when for example performance
# reaches a desired value or a specific number of epochs
# This is another good candidate for Code Kata in my opinion
# Model used is Multi-layer preceptron or DNN, no Convolution.
# Accuracy >99%,>98% on training and test is achieved in less than 5 epochs using
# Increasing epochs to 30 gave me an overfitting behaviour, adding a dropout regularization
# gave me slightly better performance on test data but gave me >99% performance on Kaggle
# test.csv, top 6%
# Network used is rather simple of one hidden latyer of 256 nodes.

import pandas as pd
import numpy as np
import tensorflow as tf

# 1- Define your callback class
# class stop_on_enough_accuracy(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs={}):
#         if(logs.get('accuracy') >= 0.99):
#             print("\nReached 99% accuracy so cancelling training!")
#             print("\nafter epoch # {0}".format(epoch))
#             self.model.stop_training = True
from tensorflow.python.keras.callbacks import ReduceLROnPlateau

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=2,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
callbacks = [learning_rate_reduction]

# 2- Load dataset
print('Loading data ...')
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print('Pre-processing data ...')
# 3- Pre-process data
x_train, x_test = x_train / 255.0, x_test / 255.0

print('Constructing model ...')
# 4- Construct model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer= tf.keras.optimizers.Ad  .Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print('Training model ...')
# 5- Train model, with callback to stop when reaching 99% accuracy
hist = model.fit(x_train, y_train,
                    epochs=30,
                    validation_data=(x_test, y_test),
                    callbacks=[callbacks], verbose=2)

print('\nAccuracy of the model on training data is {0}'
      .format(hist.history['accuracy'][-1]))

# Submit to Kaggle
# On assumption that MNIST data set loaded into keras is the same as the one
# included in the competition
# Download test.csv from
# https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/3004/861823/compressed/test.csv.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1602780773&Signature=ZxjsaQYq8LVm9d9I9Ys5iZnCQ4r33zUL1u6Il%2BGu7QDy%2BdZfvSa5uAe2dnjM4jUbf4wQuV4hVhAoxSRs51x6v2BPS6RNShhoVTUDqUWwgOYkOGwfjy3EnVr4YQGkgUG7dv8eHr0NLNkD9WZaFE06rj9FHdfzspclqNE1L9iDf7BL45uSIe5rvKSy1mSbTFuhn%2Bn4jjUbC72wSEzv2CCt529Es5BjFVdKFPx5nkIIR8JCQ6bH5NdbBB55AYmMP%2FJHtCrO%2FpNl5h2E4jSL7VVb%2BKaiNLbngIUa2minA6vF9tNsNqYadVhnUn%2FKWxHfMwyEpUM%2BuQOg1Im7wIBSLwAGtA%3D%3D&response-content-disposition=attachment%3B+filename%3Dtest.csv.zip

import matplotlib.pyplot as plt
epochs = range(len(hist.history['accuracy']))
plt.plot(epochs, hist.history['accuracy'], color = 'blue', label = 'Training')
plt.plot(epochs, hist.history['val_accuracy'], color = 'red', label = 'Validation')
plt.legend(loc='best', shadow=True)
plt.title('Training Accuracy vs Validation Accuracy')
plt.show()

test = pd.read_csv("./input/digit-recognizer/test.csv")
test = test / 255.0
test = test.values.reshape(-1, 28, 28, 1)
results = model.predict(test)
results = np.argmax(results, axis=1)
results = pd.Series(results, name='Label')
submission = pd.concat([pd.Series(range(1, 28001), name='ImageId'), results], axis=1)
submission.to_csv('mnist_kaggle_submisison.csv', index=False)
print('Done')
