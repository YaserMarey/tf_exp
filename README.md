# tf_exp
Experimenting with TensorFlow

### Experiment # 0

The simplest possible! One node neural network

### Experiment # 1

If house pricing was 50k + 50k per bedroom, so that a 1 bedroom house costs 100k,
a 2 bedroom house costs 150k etc. We want to create a neural network that learns this relationship
so that it would predict a 7 bedroom house as costing close to 400k

### Experiment # 2

Convolutional Neural Network to Detect Handwritten Digits - MINST DataSet
Convolutional Neural Network to Detect Fashion Articles - Fashion-MINST DataSet

#### DataSet
The MNIST dataset contains images of handwritten digits from 0 to 9, with 28x28 grayscale images of 65,000 fashion products from 10 categories, and 6,500 images per category. The training set has 60,000 images, and the test set has 10,000 images.
Similarly meant as a replacement to MNIST, the Fashion-MNIST dataset contains Zalando's article images, with 28x28 grayscale images of 65,000 fashion products from 10 categories, and 6,500 images per category. The training set has 60,000 images, and the test set has 10,000 images. 

#### Process
    -   Load data from keras.datasets, normalize and reshape them to x,28,28,1 and convert labels to one hot.
    -   continue the following steps using training set only, keep test set for final model verification
    -   Construct the CNN according to the architecture detailed above.
    -   Compile the model with Stochastic Gradient Descent, learning rate 0.01 and momentum is 0.9
    -   Train and cross-validate on 5 folds train, test sets for 10 epochs using a batch size of 128 sample
    -   Plot performance, diagnose, tune parameters and archicture, handle overfitting and variance.
    -   Repeat until satisfied with the model accuracy.
    -   Once done with developing the model as in the above steps:
            -   Train the model on the entire training set
            -   Test on the test set aside in the first step to verify the model performance.
    -   Save the model parameters to the disk

#### Architecture:

    -   Convolutional layer with 32 3×3 filters
    -   Pooling layer with 2×2 filter
    -   Convolutional layer with 32 3×3 filters
    -   Pooling layer with 2×2 filter
    -   Dense layer with 100 neurons
    -   Dense layer with 10 neurons, to predict the digit for the current image

#### Performance
    - Test accuracy achieved on the Never seen test set is 99.070% for MNIST
    - Test accuracy achieved on the Never seen test set is 89.95% for Fashion_MNIST

### Experiment # 3
Callback function allows us to stop the training when for example performance
reaches a desired value or a specific number of epochs.
