import numpy as np
import tensorflow as tf
import tensorflow_datasets
import tensorflow_estimator
import tensorflow_metadata
import matplotlib.pyplot as plt


layers = tf.keras.layers
print(tf.__version__)

mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train =x_train/255.0
x_test =x_test/255.0

class_names = ['T-shirt/top',
               'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


plt.figure(figsize=(10, 10))

for i in range(25):
     plt.subplot(5, 5, i+1)
     plt.xticks([])
     plt.yticks([])
     plt.grid(False)
     plt.imshow(x_train[i], cmap=plt.cm.binary)
     plt.xlabel(class_names[y_train[i]])
plt.show()

model = tf.keras.Sequential()
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',
 loss='sparse_categorical_crossentropy',
 metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20)


def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img, cmap=plt.cm.binary)
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = ‘blue’
  else:
    color = ‘red’
  plt.xlabel(“{} {:2.0f}% ({})”.format(class_names[predicted_label],
 100*np.max(predictions_array),
 class_names[true_label]),
 color=color)
def plot_value_array(i, predictions_array, true_label):
 predictions_array, true_label = predictions_array[i], true_label[i]
 plt.grid(False)
 plt.xticks([])
 plt.yticks([])
 thisplot = plt.bar(range(10), predictions_array, color=”#777777")
 plt.ylim([0, 1])
 predicted_label = np.argmax(predictions_array)
 thisplot[predicted_label].set_color(‘red’)
 thisplot[true_label].set_color(‘blue’)
 predictions = model.predict(x_test)
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, y_test, x_test)
plt.subplot(1,2,2)
plot_value_array(i, predictions, y_test)
plt.show()
