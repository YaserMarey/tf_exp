print(__doc__)

from datetime import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.tree._tree import TREE_LEAF


# Helper functions
def exp_compare_learning_algos(models_list, X_train, Y_train):
    num_folds = 10
    results = []
    names = []

    for name, model in models_list:
        kfold = KFold(n_splits=num_folds, random_state=123)
        start = time.time()
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        end = time.time()
        results.append(cv_results)
        names.append(name)
        print("%s: %f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), end - start))


def number_encode_features(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders


def plot_learning_curve(estimator, title, X, y, ylim=None,
                        cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Number of training examples")
    plt.ylabel("Accuracy Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    print("Training accuracy scores for different training sizes are:\n{0}".format(train_scores_mean))
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    print("Testing accuracy scores for different training sizes are:\n{0}".format(test_scores_mean))
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Testing score")

    plt.legend(loc="best")

    plt.show()

    return plt


def prune_index(inner_tree, index, threshold):
    if inner_tree.value[index].min() < threshold:
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
    # if there are shildren, visit them as well
    if inner_tree.children_left[index] != TREE_LEAF:
        prune_index(inner_tree, inner_tree.children_left[index], threshold)
        prune_index(inner_tree, inner_tree.children_right[index], threshold)


def plot_validation_curve(estimator=None, title=None, X=None, y=None, ylim=None,
                          cv=None, n_jobs=None, param=None, param_values=None,
                          train_scores=None, test_scores=None):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(param)
    plt.ylabel("Accuracy Score")
    if estimator != None:
        train_scores, test_scores = validation_curve(estimator, X, y, param, param_values, cv=cv, n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    print("Training accuracy scores for different training sizes are:\n{0}".format(train_scores_mean))
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    print("Testing accuracy scores for different training sizes are:\n{0}".format(test_scores_mean))
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(param_values, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(param_values, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(param_values, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(param_values, test_scores_mean, 'o-', color="g",
             label="Testing score")

    plt.legend(loc="best")

    plt.show()

    return plt
