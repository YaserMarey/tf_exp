#!/usr/bin/env python
# This is my first submission to Kaggle. I am going first to lightly explore the training dataset,
# then following that I am going to clean it, interpolate missing values, and convert categorical
# values into numerical, and then train a DNN using Keras.
# I am assuming that training set and test set are coming have the same distribution therefore,
# once I am settled on the best model from working on training I am applying the same data
# pre-processing steps to test data set before passing to the model.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm as cm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import util

def explore_ds(data, label_col_name):

    print("Total Number of Samples: {0}\nNumber of attributes: {1}".format(data.shape[0], data.shape[1] + 1))

    print("Number of Not survived Records: {0} accounts for: {1:.2f}% of the survived class\n"
          "Number of Survived Records: {2} accounts for: {3:.2f}% of the survived class".format(
        data.loc[data[label_col_name] == 0].shape[0],
        100 * data.loc[data[label_col_name] == 0].shape[0] / data.shape[0],
        data.loc[data[label_col_name] == 1].shape[0],
        100 * data.loc[data[label_col_name] == 1].shape[0] / data.shape[0]))
    print("Here's the data types of our columns:\n", data.dtypes)
    print(data.describe())
    # Check data set imbalance
    unique, counts = np.unique(data[label_col_name], return_counts=True)
    plt.bar(unique, counts, 1, color=['Green', 'Red'])
    plt.title('Class Frequency')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.show()
    # Check null values
    print(data.apply(lambda x: x.isnull().values.ravel().sum()))
    print(data.apply(lambda x: x.isna().values.ravel().sum()))

    # Plot distribution
    import seaborn as sns
    data.plot(kind='density', subplots=True, layout=(5, 7), sharex=False, legend=False, fontsize=1)
    plt.show()
    plt.figure(figsize=(12, 12))

    corr = data.corr()

    ax = sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )

    plt.show()

def preprocess_ds(data):
    # Drop unneeded columns
    data.drop("PassengerId", axis=1, inplace=True)
    data.drop('Cabin', axis=1, inplace=True)
    data.drop("Name", axis=1, inplace=True)
    data.drop("Ticket", axis=1, inplace=True)

    # Interpolate missing age values with mean age of the same gender
    data['Age'] = data['Age'].fillna(data.groupby('Sex')['Age'].transform('mean'))

    # Replace missing Embarked values with S since it the most common Embarked value
    data.fillna({"Embarked": "S"}, inplace=True)

    # Map the categorical column 'Sex' with numerical data
    sex_mapping = {"male": 0, "female": 1}
    data['Sex'] = data['Sex'].map(sex_mapping)

    # Map the categorical column 'embarked' with numerical data
    embarked_mapping = {"S": 1, "C": 2, "Q": 3}
    data['Embarked'] = data['Embarked'].map(embarked_mapping)

    return {'data': data.drop(["Survived"], axis=1), 'target':data["Survived"]}

def analyze__ds(ds):
    X, y = ds['data'], ds['target']

    plot_learning_ds(X, y)

    plot_model_complexity_curves_ds(X, y)


def plot_learning_ds(X, y):
    # BC - Exp - 1 - Decision Tree
    title = "Learning Curve for Decision Tree run on Breast Cancer DS\n" \
            "Cross Validation, averaged over: 100 rounds,\n" \
            "Training-Test: 80-20%"
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    estimator = DecisionTreeClassifier()
    util.plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

    # BC - Exp - 1 - Multilayer Preceptron Network
    title = "Learning Curve for Multi Layer Preceptron (MLP) on Breast Cancer DS\n" \
            "Cross Validation, averaged over: 10 rounds,\n" \
            "Training-Test: 80-20%"
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    estimator = MLPClassifier(random_state=42)
    util.plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)

    # BC - Exp - 1 - Adaboost
    title = "Learning Curve for Adaboost run on Breast Cancer DS\n" \
            "Cross Validation, averaged over: 10 rounds,\n" \
            "Training-Test: 80-20%, n_estimators: 50, learning_rate:1)"
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    estimator = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=0)
    util.plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)


    # BC - Exp - 1 - K-Nearest Neighbor
    title = "Learning Curves for K-Nearest Neighbor (KNN) on Breast Cancer DS\n" \
            "Cross Validation, averaged over 10 rounds,\n" \
            "Training-Test:80-20%, n_neighbors:5"
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    estimator = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    util.plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)

    # CEN - Exp - 1 - rbf Kernel - Support Vector Machines
    title = "Learning Curves for Support Vector Machines (SVM) on Breast Cancer DS\n" \
            "Cross Validation, averaged over 10 rounds,\n" \
            "Training-Test:80-20%, kernel: rbf, gamma:0.001"
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    estimator = SVC(kernel='rbf', gamma=0.001, random_state=0)
    util.plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)

    # CEN - Exp - 2 - linear Kernel - Support Vector Machines
    title = "Learning Curves for Support Vector Machines (SVM) on Breast Cancer DS\n" \
            "Cross Validation, averaged over 10 rounds,\n" \
            "Training-Test:80-20%, kernel: linear, gamma:0.001"
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    estimator = SVC(kernel='linear', gamma=0.001, random_state=0)
    util.plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)


def plot_model_complexity_curves_ds(X, y):
    # BC - Exp - 2 - Decision Tree
    title = "Pruning by Limiting Max Depth of Decision Tree run on Breast Cancer DS\n" \
            "Cross Validation, averaged over: 100 rounds,\n" \
            "Training-Test: 80-20%"
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    estimator = DecisionTreeClassifier()
    util.plot_validation_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4, param='max_depth',
                               param_values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # BC - Exp - 3 - Decision Tree
    title = "Complexity Analysis of Decision Tree run on Breast Cancer DS\n" \
            "Cross Validation, averaged over: 100 rounds,\n" \
            "Training-Test: 80-20%"
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    estimator = DecisionTreeClassifier()
    util.plot_validation_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4, param='min_samples_leaf',
                               param_values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # BC - Exp - 2 - Multilayer Preceptron Network
    title = "Complexity Analysis for Multi Layer Preceptron (MLP) on Breast Cancer DS\n" \
            "Cross Validation, averaged over: 10 rounds,\n" \
            "Training-Test: 80-20%"
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    estimator = MLPClassifier(random_state=42)
    util.plot_validation_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4, param='alpha',
                               param_values=np.logspace(-5, 0.5, 20))


    # BC - Exp - 2 - Adaboost
    title = "Pruning by Limiting Max Depth of Decision Tree Based Adaboost\n" \
            "run on Breast Cancer DS, Cross Validation, avg: 10 rounds\n" \
            "Training-Test: 80-20%, learning_rate:1)"
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    param_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    train_scores = list()
    test_scores = list()
    for param in param_values:
        estimator = AdaBoostClassifier(DecisionTreeClassifier(max_depth=param), n_estimators=50, learning_rate=1, random_state=0)
        this_scores_dict = cross_validate(estimator, X, y, cv=cv, return_train_score=True)
        train_scores.append(this_scores_dict['train_score'])
        test_scores.append(this_scores_dict['test_score'])
    util.plot_validation_curve(title=title, ylim=(0.7, 1.01), param='max_depth', param_values=param_values,train_scores=train_scores, test_scores=test_scores)

    # BC - Exp - 3 - Adaboost
    title = "Complexity Analysis for Adaboost run on Breast Cancer DS\n" \
            "Cross Validation, averaged over: 10 rounds,\n" \
            "Training-Test: 80-20%, learning_rate:1)"
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    estimator = AdaBoostClassifier(learning_rate=1, random_state=0)
    util.plot_validation_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4, param='n_estimators',
                               param_values=[50,100,150,200,250,300,350,400])

    # BC - Exp - 2 - K-Nearest Neighbor
    title = "Complexity Analysis for K-Nearest Neighbor (KNN) on Breast Cancer DS\n" \
            "Cross Validation, averaged over 10 rounds,\n" \
            "Training-Test:80-20%"
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    estimator = KNeighborsClassifier(metric='minkowski', p=2)
    util.plot_validation_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4, param='n_neighbors',
                               param_values=[1, 3, 5, 7, 9, 11, 13, 15])

    # BC - Exp - 2 -  SVC
    title = "Complexity Analysis for Support Vector Machines (SVM) on BC DS\n" \
            "Cross Validation, averaged over 10 rounds,\n" \
            "Training-Test:80-20%, kernel: linear, gamma:0.001"
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = SVC(kernel='linear', gamma=0.001, random_state=0)
    util.plot_validation_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4, param='C', param_values=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0])

#######################################################################################################################
def main():
    ######################
    #
    ######################
    # Step 0: Read Data from File
    data = pd.read_csv('train.csv')

    # Step 1: Explore Data Set
    # explore_ds(data, 'Survived')
    # Step 2: Preprocess Data
    ds = preprocess_ds(data)

    # Step 3: Data Set Analysis
    analyze__ds(ds)

    # Step 4: Generate Submission
    # generate_submission()

def generate_submission():
    global test_data, ids, predictions
    # Prediction
    test_data = pd.read_csv('test.csv')
    test_data.drop('Cabin', axis=1, inplace=True)
    test_data.drop("Name", axis=1, inplace=True)
    test_data.drop("Ticket", axis=1, inplace=True)
    # #
    test_data['Age'] = test_data['Age'].fillna(train_data.groupby('Sex')['Age'].transform('mean'))
    test_data.fillna({"Embarked": "S"}, inplace=True)
    test_data['Sex'] = test_data['Sex'].map(sex_mapping)
    test_data['Embarked'] = test_data['Embarked'].map(embarked_mapping)
    test_data['Fare'].fillna(np.mean(test_data['Fare']), inplace=True)
    ids = test_data['PassengerId']
    predictions = model.predict(test_data.drop('PassengerId', axis=1))
    # Preparing Kaggle submission file
    predictions = predictions.reshape((predictions.shape[0],))
    predictions = np.where(predictions >= 0.5, 1, 0)
    output = pd.DataFrame({'PassengerId': ids, 'Survived': predictions})
    output.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
#

# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# construct model
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(128, input_shape=(7,), activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# 
# model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# x_train = np.array(x_train).astype('float32')
# y_train = np.array(y_train).astype('int32').reshape(-1, 1)

#
# print('\n Training .... ')
# hist = model.fit(x_train, y_train,
#                  batch_size=256,
#                  epochs=500,
#                  verbose=0,
#                  validation_split=0.2)
#
# print('\n accuracy after last training epoch | {0} |'.format(hist.history['val_accuracy'][-1]))
# print('\n Evaluation .... ')
# results = model.evaluate(x_test, y_test, verbose=0)
# print('\n accuracy of evaluation on test data non seen before | {0} |'.format(results[1]))
#
# # -----------------------------------------------------------
# # Retrieve a list of list results on training and test data
# # sets for each training epoch
# # -----------------------------------------------------------
# acc = hist.history['accuracy']
# val_acc = hist.history['val_accuracy']
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']
#
# epochs = range(len(acc))  # Get number of epochs
#
# # ------------------------------------------------
# # Plot training and validation accuracy per epoch
# # ------------------------------------------------
# plt.plot(epochs, acc)
# plt.plot(epochs, val_acc)
# plt.title('Training and validation accuracy')
# plt.figure()
#
# # ------------------------------------------------
# # Plot training and validation loss per epoch
# # ------------------------------------------------
# plt.plot(epochs, loss)
# plt.plot(epochs, val_loss)
# plt.title('Training and validation loss')
#
# plt.show()

