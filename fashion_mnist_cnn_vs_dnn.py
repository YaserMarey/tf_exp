import tensorflow as tf
# Exp - 5
# Comparing CNN and DNN applied to Fahsion-MNIST

results = []

# First CNN
# 1- Load data
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 2- Pre-process data
x_train=x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
x_train, x_test= x_train / 255.0, x_test / 255.0

# 3- Construct model
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.summary()

# 4- Train model
history = model.fit(x_train, y_train, epochs=5, verbose=0)

# 5- Evaluate model
test_loss = model.evaluate(x_test, y_test, verbose=0)

# Collect training accuracy
results.append(history.history['accuracy'][-1])
# Collect test accuracy
results.append(test_loss[1])

# Second DNN

# 1- Load dataset
# print('Loading data ...')
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
# print('Pre-processing data ...')
# 2- Pre-process data
x_train, x_test = x_train / 255.0, x_test / 255.0
# print('Constructing model ...')
# 4- Construct model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(265, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# print('Training model ...')
# 5- Train model, with callback to stop when reaching 99% accuracy
history = model.fit(x_train, y_train, epochs=10, verbose=0)
results.append(history.history['accuracy'][-1])
# print('Evaluating model ...')
# 6- Evaluate model on unseen test data
test_loss = model.evaluate(x_test, y_test, verbose=0)
results.append(test_loss[1])

print('CNN results')
print('\nAccuracy of the model on training data is {0}'.format(results[0]))
print('\nAccuracy of the model on unseen test data is {0}'.format(results[1]))
print('DNN results')
print('\nAccuracy of the model on training data is {0}'.format(results[2]))
print('\nAccuracy of the model on unseen test data is {0}'.format(results[3]))
