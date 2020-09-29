import numpy as np
import tensorflow as tf
import tensorflow_datasets
import tensorflow_estimator
import tensorflow_metadata
import matplotlib.pyplot as plt

# import tensorflow as tf
# tf.get_logger().setLevel('ERROR')
#
#
# x = [[2.]]
# print('Tensorflow Version ', tf.__version__)
# print('hello TF world, {}'.format(tf.matmul(x, x)))



# layers = tf.keras.layers
# print(tf.__version__)

# mnist = tf.keras.datasets.fashion_mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train =x_train/255.0
# x_test =x_test/255.0

# class_names = ['T-shirt/top',
#                'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
#                'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#
# # plt.figure(figsize=(10, 10))
#
# # for i in range(25):
# #      plt.subplot(5, 5, i+1)
# #      plt.xticks([])
# #      plt.yticks([])
# #      plt.grid(False)
# #      plt.imshow(x_train[i], cmap=plt.cm.binary)
# #      plt.xlabel(class_names[y_train[i]])
# #
# # plt.show()
#
# model = tf.keras.Sequential()
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))
# model.compile(optimizer='adam',
#  loss='sparse_categorical_crossentropy',
#  metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=5)


