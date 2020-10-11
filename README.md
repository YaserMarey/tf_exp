# tf_exp
Experimenting with TensorFlow

### Experiment # 0

The simplest possible! One node neural network

### Experiment # 1

If house pricing was 50k + 50k per bedroom so that a 1 bedroom house costs 100k,
a 2 bedroom house costs 150k, etc. We want to create a neural network that learns this relationship
so that it would predict a 7 bedroom house as costing close to 400k

### Experiment # 2
In this experiment, I am developing a template like code that has the needed steps to train and classify images. 
I am testing this template with two datasets and I am using CNN as the following:

Convolutional Neural Network to Detect Handwritten Digits - MINST DataSet 
http://yann.lecun.com/exdb/mnist/ 

Convolutional Neural Network to Detect Fashion Articles - Fashion-MINST DataSet https://github.com/zalandoresearch/fashion-mnist

#### DataSet
The MNIST dataset contains images of handwritten digits from 0 to 9, with 28x28 grayscale images of 65,000 fashion products from 10 categories, and 6,500 images per category. The training set has 60,000 images, and the test set has 10,000 images.
Similarly meant as a replacement to MNIST, the Fashion-MNIST dataset contains Zalando's article images, with 28x28 grayscale images of 65,000 fashion products from 10 categories, and 6,500 images per category. The training set has 60,000 images, and the test set has 10,000 images. 

#### Process
    -   Load data from Keras.datasets, normalize, and reshape them to x,28,28,1, and convert labels to one hot.
    -   continue the following steps using the training set only, keep test set for final model verification
    -   Construct the CNN according to the architecture detailed above.
    -   Compile the model with Stochastic Gradient Descent, learning rate 0.01 and momentum is 0.9
    -   Train and cross-validate on 5 folds train, test sets for 10 epochs using a batch size of 128 sample
    -   Plot performance, diagnose, tune parameters and architecture, handle overfitting and variance.
    -   Repeat until satisfied with the model accuracy.
    -   Once done with developing the model as in the above steps:
            -   Train the model on the entire training set
            -   Test on the test set aside in the first step to verify the model performance.
    -   Save the model parameters to the disk

#### Architecture:
I am using the same architecture for both datasets and interestingly it works satisfactory enough without a change from MNIST to Fashion-MNIST datasets:

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
The callback function allows us to stop the training when for example performance
reaches the desired value or a specific number of epochs.

### Experiment # 4
Comparing CNN and DNN applied to Fashion-MNIST. This is meant to compare implementation details rather than performance.
Tensorflow and Keras make implementation very similar by hiding most of the details. tf.keras.layers provide simple Conv2D and MaxPooling2D to create a convolution and max-pooling layers.
##### CNN results
-   Accuracy of the model on training data is 0.9301833510398865
-   Accuracy of the model on unseen test data is 0.9056000113487244
##### DNN results
-   Accuracy of the model on training data is 0.9962666630744934
-   Accuracy of the model on unseen test data is 0.9800999760627747


### Experiment # 5
CNN that classify images to happy and sad Constructed with 3 convolutional layers.
Dataset is save to h-or-s folder on the disk. this folder contains two sub-folders '/h-or-s/happy and '/h-or-s/sad' each contains 30 images of happy or sad faces respectively.
There is also a separate folder for validation images '/validation-h-or-s' which again contains two subfolders '/validation-h-or-s/happy' '/validation-h-or-s/sad' each contains 10 images.
The code reads the images from the disk instead of loading them from Keras as we did before with MNIST and Fashion-MNIST. 
labels are inferred from folders' names

### Experiment # 6
This is my first submission to Kaggle. I am going first to lightly explore the training dataset, then following that I am going to clean it, interpolate missing values, and convert categorical
values into numerical, and then train a DNN using Keras.
I am assuming that training set and test set are coming have the same distribution therefore, once I am settled on the best model from working on training I am applying the same data pre-processing steps to test data set before passing to the model.

# Experiment - 7
CNN applied to Kaggle Cats vs Dogs dataset. Trying to optimize the CNN Architecture by clearly monitoring overfitting 
levles of the network.

# Experiment - 8
Pre-trained NN models offer a great opportunity to build on other researchers work.
Inception which is a Deep Learning Convolutional Architecture presented by 
[Szegedy et al 2014](http://https://arxiv.org/abs/1409.4842) is an important milestone in the development of CNN 
classifiers.

In this Notebook I am applying a pre-trained model of Inception version 3 on the welknown Kaggle dataset of Cats vs Dogs. 
Inception V3 was trained using a dataset of 1,000 classes, including Cats and Dogs, from the original ImageNet dataset 
which was trained with over 1 million training images, the Tensorflow version has 1,001 classes which is due to 
an additional "background' class not used in the original ImageNet.

# Experiment - 9
DNN applied to Kaggle Cats vs Dogs dataset, images are read using ImageDataGenerator,and I manipulated the data set 
so that Cross Validation is used in training.
Image augmentation is applied as well. I am also using the final NN architecture I concluded Exp 7. 

