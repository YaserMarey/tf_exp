# Exp - 5
# CNN that classify images to happy and sad
# Constructed with 3 convolutional layers.
# Dataset is save to h-or-s folder on the disk. this folder contains two
# sub-folders '/h-or-s/happy and '/h-or-s/sad' each contains 30 images of happy or sad faces
# respectively.
# There is also a separate folder for validation images '/validation-h-or-s' which again
# contains two subfolders '/validation-h-or-s/happy' '/validation-h-or-s/sad' each contains 10 images
# The code reads the images from the disk instead of loading them from Keras as we did before
# with MNIST and Fashion-MNIST. labels are inferred from folders' names

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DESIRED_ACCURACY = 0.999
class myCallback(tf.keras.callbacks.Callback):
    # Your Code
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') >= DESIRED_ACCURACY):
            print('\nStopped after reaching desired accuracy!')
            self.model.stop_training = True

callbacks = myCallback()

# This Code Block should Define and Compile the Model. Please assume the images are 150 X 150 in your implementation.
model = tf.keras.models.Sequential([
    # Your Code Here
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Here we create ImageDataGenerator and we normalzie while loading
train_datagen = ImageDataGenerator(rescale=1 / 255)
validation_datagen = ImageDataGenerator(rescale=1/255)

# We then load data through the generator
train_generator = train_datagen.flow_from_directory(
    './h-or-s',
    target_size=(150, 150), # Resize the image while loading
    batch_size=15,#
    class_mode='binary') # 1 Dimensional binary labels

validation_generator = validation_datagen.flow_from_directory(
        './validation-h-or-s',  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=10,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')


# here we train the model
model.fit_generator(
    train_generator,
    epochs=15,# Pass the entire dataset for 5 times, seems like it will overfit!
    verbose=1,
    callbacks=[callbacks],
    validation_data=validation_generator)



