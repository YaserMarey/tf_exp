# tf_exp
Experimenting with TensorFlow

Experiment # 1


Convlutional Nerual Network to Detect Handwritten Digits - MINST DataSet

####DataSet
The Fashion-MNIST dataset contains Zalando's article images, with 28x28 grayscale images of 65,000 fashion products from 10 categories, and 6,500 images per category. The training set has 55,000 images, and the test set has 10,000 images. 

####Architecture:

    -   Convolutional layer with 32 5×5 filters
    -   Pooling layer with 2×2 filter
    -   Convolutional layer with 64 5×5 filters
    -   Pooling layer with 2×2 filter
    -   Dense layer with 1024 neurons
    -   Dense layer with 10 neurons, to predict the digit for the current image

Process:

    Build the input layer using the reshape() function.
    Build the convolutional/pooling layers using the layers.conv2d()  and layers.max_pooling2d() functions.
    Build the dense layers using the layers.dense() function.
    Generate predictions by running the softmax() function.
    Calculate loss by running the losses.sparse_softmax_cross_entropy() function.
    Configure the training operation using the optimizer.minimize() function.
    Add an evaluation metric using tf.metrics.accuracy()
    Load data using the mnist.load_data() function.
    Define an Estimator for the custom object detection model (the example provides a ready-made estimator for MNIST data).
    Train the model by running the train() function on the Estimator object.
    Evaluate the model on all MNIST images using the evaluate() function.
    