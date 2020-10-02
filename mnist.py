# Exp-2
# ### Experiment # 2
# In this experiment I am developing a template like code that has the needed steps to train and classify images.
# I am testing this template with two datasets:
# Convolutional Neural Network to Detect Handwritten Digits - MINST DataSet http://yann.lecun.com/exdb/mnist/
# Convolutional Neural Network to Detect Fashion Articles - Fashion-MINST DataSet https://github.com/zalandoresearch/fashion-mnist
#
# #### DataSet
# The MNIST dataset contains images of handwritten digits from 0 to 9, with 28x28 grayscale images of 65,000 fashion products from 10 categories, and 6,500 images per category. The training set has 60,000 images, and the test set has 10,000 images.
# Similarly meant as a replacement to MNIST, the Fashion-MNIST dataset contains Zalando's article images, with 28x28 grayscale images of 65,000 fashion products from 10 categories, and 6,500 images per category. The training set has 60,000 images, and the test set has 10,000 images.
#
# #### Process
#     -   Load data from keras.datasets, normalize and reshape them to x,28,28,1 and convert labels to one hot.
#     -   continue the following steps using training set only, keep test set for final model verification
#     -   Construct the CNN according to the architecture detailed above.
#     -   Compile the model with Stochastic Gradient Descent, learning rate 0.01 and momentum is 0.9
#     -   Train and cross-validate on 5 folds train, test sets for 10 epochs using a batch size of 128 sample
#     -   Plot performance, diagnose, tune parameters and archicture, handle overfitting and variance.
#     -   Repeat until satisfied with the model accuracy.
#     -   Once done with developing the model as in the above steps:
#             -   Train the model on the entire training set
#             -   Test on the test set aside in the first step to verify the model performance.
#     -   Save the model parameters to the disk
#
# #### Architecture:
# I am using same architecture for both datasets and interestingly it works staifacotry enough without a change from MNIST to Fashion-MNIST datasets:
#
#     -   Convolutional layer with 32 3×3 filters
#     -   Pooling layer with 2×2 filter
#     -   Convolutional layer with 32 3×3 filters
#     -   Pooling layer with 2×2 filter
#     -   Dense layer with 100 neurons
#     -   Dense layer with 10 neurons, to predict the digit for the current image

import numpy as np
import sklearn.model_selection as sklrn
import tensorflow.keras as k
from matplotlib import pyplot


# The process of finding the best model passes through two phases:

# Phase I: We develop the model, by loading data, pre-process it, select architecture and different parameters
# train and test using cross-validation, diagnose, adjust and repeat until you are satisfied with the performance

# This is what is done by the following develop_the_model() function.

# Phase II: We come out of phase I with the right model we want to keep,
# now we fit the model to the entire training set and save the model parameters to the disk

# This is what is done by the save_final_model()

#  Here is first Phase - I
def develop_the_model():
    # Step - 1
    # Load data, Keras already supports API interface for a number of
    # wellknown data sets including MNIST.
    # This interface allows us to load data as the following:
    x_train, y_train, x_test, y_test = load_dataset()

    # Step - 2
    # Reshape, normalize and/or standardize data
    x_train, y_train, x_test, y_test = prepare_data(x_train, y_train, x_test, y_test)

    # Step - 3
    # develop the model, that is we try to find best parameters by training the model
    # and cross-validation
    n_folds, epochs, batch_size = 5, 10, 128
    model = construct_and_compile_model()
    scores, histories = train_and_cross_validate(model, x_train, y_train, n_folds, epochs, batch_size)

    # Step - 4 - A
    # Diagnose, by observing the leanring curves over epochs for both training and validation for
    # different (train, test) folds
    plot_learning_curves(histories)

    # Step - 4 - B
    # summarize performance on test measured by the accuracy at the end
    # of the epochs of each (train, test) fold. Summary is presneted as boxplot, and also
    # mean, and standard deviation of all measured accuracies over folds
    summarize_performance(scores)
    return model, x_train, y_train, x_test, y_test, epochs, batch_size


#  Then Phase - II
def save_final_model(model, x_train, y_train, epochs, batch_size, filename):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    model.save(filename)


def evaluate_final_model(model, x_test, y_test):
    # evaluate model on test dataset
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Final model accuracy > %.3f' % (acc * 100.0))
    return (acc * 100.0)

# ###########################################

def load_dataset():
    # (x_train, y_train), (x_test, y_test) = k.datasets.mnist.load_data()
    (x_train, y_train), (x_test, y_test) = k.datasets.fashion_mnist.load_data()
    # summarize loaded dataset
    print('Train: X={0}, y={1}'.format(x_train.shape, y_train.shape))
    print('Test: X={0}, y={1}'.format(x_test.shape, y_test.shape))
    # plot_first_few_images(x_train)
    return x_train, y_train, x_test, y_test


def prepare_data(x_train, y_train, x_test, y_test):
    # Normalize data by dividing on maximum value which makes data values to come in the range [0,1]
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    # Reshape inputs to to 28,28,1 dimensions
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    # Convert class vectors to one hot encoded values
    y_train = k.utils.to_categorical(y_train, 10)
    y_test = k.utils.to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test


def construct_and_compile_model():
    model = k.Sequential(
        [
            k.Input(shape=(28, 28, 1)),
            k.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", kernel_initializer='he_uniform'),
            k.layers.MaxPooling2D(pool_size=(2, 2)),
            k.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", kernel_initializer='he_uniform'),
            k.layers.MaxPooling2D(pool_size=(2, 2)),
            k.layers.Flatten(),
            k.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'),
            k.layers.Dropout(0.4),  # add this in Fashion-MNIST Case to regularize CNN
            k.layers.Dense(10, activation="softmax")
        ]
    )
    model.compile(loss=k.losses.categorical_crossentropy,
                  optimizer=k.optimizers.SGD(lr=0.01, momentum=0.9),
                  metrics=['accuracy'])

    return model


def train_and_cross_validate(model, x_data, y_data, n_folds, epochs, batch_size):
    scores, histories = [], []
    # prepare cross validation
    kfold = sklrn.KFold(n_folds, shuffle=True, random_state=1)
    # enumerate splits
    for train_ix, test_ix in kfold.split(x_data):
        # select rows for train and test
        xx_train, yy_train, xx_test, yy_test = \
            x_data[train_ix], y_data[train_ix], x_data[test_ix], y_data[test_ix]
        # fit model = train the model
        history = model.fit(xx_train,
                            yy_train,
                            epochs=epochs,  # The more we train the more our model fits the data
                            batch_size=batch_size,  # Smaller batch sizes = samller steps towards convergence
                            validation_data=(xx_test, yy_test),
                            verbose=0)
        # evaluate model
        _, accuracy = model.evaluate(xx_test, yy_test, verbose=0)

        print('> %.3f' % (accuracy * 100.0))

        # stores scores
        scores.append(accuracy)
        histories.append(history)
    return scores, histories


# plot learning curves
def plot_learning_curves(histories):
    for i in range(len(histories)):
        # plot loss
        pyplot.subplot(2, 1, 1)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(histories[i].history['loss'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')

        # plot accuracy
        pyplot.subplot(2, 1, 2)
        pyplot.title('Classification Accuracy')
        pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    pyplot.show()


# box plot summary of model performance
def summarize_performance(scores):
    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (np.mean(scores) * 100, np.std(scores) * 100, len(scores)))
    # box and whisker plots of results
    pyplot.boxplot(scores)
    pyplot.show()


# Utils
def plot_first_few_images(x_train):
    # plot first few images
    for i in range(9):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # plot raw pixel data
        pyplot.imshow(x_train[i], cmap=pyplot.get_cmap('gray'))
    # show the figure
    pyplot.show()

# Call to Phase I: Develpe the model
model, x_train, y_train, x_test, y_test, epochs, batch_size = develop_the_model()

# Call to Phase II: save_final_model() and evaluate_final_model()
save_final_model(model, x_train, y_train, epochs, batch_size, 'fashion_mnist_final_model.h5')
evaluate_final_model(model, x_test, y_test)
