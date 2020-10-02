# Exp-3
# Callback function allows us to stop the training when for example performance
# reaches a desired value or a specific number of epochs
# This is another good candidate for Code Kata in my opinion
# Model used is Multi-layer preceptron or DNN, no Convolution.
# Accuracy >99%,>98% on training and test is achieved in less than 5 epochs using two layers DNN

import tensorflow as tf

# 1- Define your callback class
class stop_on_enough_accuracy(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') >= 0.99):
            print("\nReached 99% accuracy so cancelling training!")
            print("\nafter epoch # {0}".format(epoch))
            self.model.stop_training = True

# 2- Load dataset
print('Loading data ...')
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
print('Pre-processing data ...')
# 3- Pre-process data
x_train, x_test = x_train / 255.0, x_test / 255.0
print('Constructing model ...')
# 4- Construct model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(265, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print('Training model ...')
# 5- Train model, with callback to stop when reaching 99% accuracy
callbacks = stop_on_enough_accuracy()
history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks], verbose=0)
print('\nAccuracy of the model on training data is {0}'.format(history.history['accuracy'][-1]))

print('Evaluating model ...')
# 6- Evaluate model on unseen test data
test_loss = model.evaluate(x_test, y_test, verbose=0)
print('\nAccuracy of the model on unseen test data is {0}'.format(test_loss[1]))


