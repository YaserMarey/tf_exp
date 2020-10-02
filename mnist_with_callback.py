# Exp-3
# Callback function allows us to stop the training when for example performance
# reaches a desired value or a specific number of epochs
# This is another good candidate for Code Kata in my opinion

import tensorflow as tf

# 1- Define your callback class
class stop_on_enough_accuracy(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') >= 0.99):
            print("\nReached 99% accuracy so cancelling training!")
            print("\nat epoch # {0}".format(epoch))
            self.model.stop_trainig = True

# 2- Load dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# 3- Pre-process data
x_train, x_test = x_train / 255.0, x_test / 255.0

# 4- Construct model

# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(512, activation=tf.nn.relu),
#   tf.keras.layers.Dense(10, activation=tf.nn.softmax)
# ])

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(32, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 5- Train model, with callback to stop when reaching 99% accuracy

callbacks = stop_on_enough_accuracy()
model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])



