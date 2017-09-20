#!/usr/bin/env python
"""This file contains the Linear Regression Exercise 1 from
Andrew Ng Coursera course
https://www.coursera.org/learn/machine-learning/home/welcome
buy using Python instead Octave.
"""
from os import path
import random
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt
from scipy.optimize import minimize
from scipy.io import loadmat

from aiml import *

CHECKPOINT_FILE = 'cache_%s' % path.splitext(path.basename(__file__))[0]


def multiclass_logistic_regression_k_problems(checkpoint=True):
    "Make a simple linear regression from from Andre Ng Exercise 1"
    chp = Checkpoint(CHECKPOINT_FILE + '.multiclass')

    # Load data from CSV
    data = loadmat('ex3data1.mat')

    # X: 5000 samples x 400 pixels per image (aka features)
    # y: 5000 class belonging vector from 1 to 10 (1-index based!)
    # so in this example, K=10
    X, y = data['X'], data['y']

    # just take a simple random sample and draw it
    # selected = random.randint(0, X.shape[0])
    # image = X[selected].reshape((20, 20))
    # plt.imshow(image, cmap='gray')
    # plt.title('Should be a %s' % y[selected])
    # plt.show()

    # setup optimization params
    klasses = np.unique(y)
    n_klasses = klasses.size
    samples, features = X.shape

    # Theta is now a matrix of (klasses x features)
    # don't forget the bias term for each klass
    Theta = np.random.randn(1 + features, n_klasses)
    Theta[0] = 1  # the bias term
    H = np.insert(X, 0, values=1, axis=len(X.shape) > 1)  # axis 0 or 1
    y.shape = (samples, )

    # we solve (k) simple problems, using different
    # Theta columns and building a custom y vector for
    # each problem
    learning_rate = 0.1

    # check if we have solved the problem before, just for fast debugging
    if checkpoint and os.path.exists(CHECKPOINT_FILE):
        Theta = chp.load()['Theta']
    else:
        for k, klass in enumerate(klasses):
            y_i = (y == (k + 1)) * 1
            theta = Theta[:, k]
            theta, result = solve_logistic_regression(
                H, y_i, theta, learning_rate)
            Theta[:, k] = theta
            print "Class %s : error: %s, %s" % (k, result.fun, result.message)
        chp.save(Theta=Theta)

    # Train accuracy
    p = predict_multi_class(Theta, H)
    p += 1   # y is 1-based index in the examples!
    print "Train Accuracy: %f" % (p == y).mean()


def multiclass_logistic_regression_single_problem(checkpoint=True):
    """Make a logistic regression building a single problem
    and trying to solve as a whole.

    checkpoint: if True, try to load the Theta solution from previous run.
    """
    chp = Checkpoint(CHECKPOINT_FILE + '.singleclass')

    def rebuilt(X, theta):
        """Reshape the Theta tensor in single vector suitable for
        optimization algorithms"""
        features = X.shape[1]
        klasses = theta.size // features

        assert features * klasses == theta.size  # perfect fitting

        Theta = np.reshape(theta, (features, klasses))
        return Theta


    def cost(theta, X, Y, lamb=0):
        "Compute the cost of logistic regression"
        H = np.insert(X, 0, values=1, axis=len(X.shape) > 1)  # axis 0 or 1

        Theta = rebuilt(H, theta)
        h = sigmoid(np.dot(H, Theta))

        cost = -Y * np.log(h) - (1 - Y) * np.log(1 - h)
        cost = cost.sum()
        if lamb > 0:  # regularization ignore bias parameter
            theta2 = np.copy(theta)
            theta2[0] = 0
            reg = lamb * np.dot(theta2, theta2) / 2.0
            cost += reg

        cost /= Y.shape[0]
        return cost


    def grad(theta, X, Y, lamb=0):
        "Compute the gradient of logistic regression."
        X = np.insert(X, 0, values=1, axis=len(X.shape) > 1)  # axis 0 or 1

        n = Y.shape[0]

        Theta = rebuilt(X, theta)
        H = sigmoid(np.dot(X, Theta))

        d = H - Y
        grad = X.T.dot(d)

        if lamb > 0:  # regularization ignore bias parameter
            # theta2 = np.zeros_like(theta)
            # theta2[1:] = theta[1:]
            Theta2 = np.copy(Theta)
            Theta2[0] = 0
            reg = lamb * Theta2
            grad += reg

        grad /= n

        # restore same shape that minimization algorithms uses
        grad.shape = theta.shape
        return grad


    # Load data from CSV
    data = loadmat('ex3data1.mat')

    # X: 5000 samples x 400 pixels per image (aka features)
    # y: 5000 class belonging vector from 1 to 10 (1-index based!)
    # so in this example, K=10
    X, y = data['X'], data['y']

    # setup optimization params
    klasses = np.unique(y)
    n_klasses = klasses.size
    samples, features = X.shape

    # Theta is now a matrix of (klasses x features)
    # don't forget the bias term for each klass
    Theta = np.random.randn(1 + features, n_klasses)
    Theta[0] = 1  # the bias term
    H = np.insert(X, 0, values=1, axis=len(X.shape) > 1)  # axis 0 or 1
    y.shape = (samples, )

    # setup the Y multiclass matrix
    Y = np.zeros((samples, n_klasses), dtype=np.int8)
    for i, klass in enumerate(y):
        Y[i][klass - 1] = 1

    learning_rate = 0.1

    # save theta evolution
    trajectory = []

    def retail(theta):
        "save each step of the training"
        # Theta = rebuilt(X, theta)
        # p = predict_multi_class(Theta, X)
        # p += 1   # y is 1-based index in the examples!
        # print "Train Accuracy: %f" % (p == y).mean()
        trajectory.append(theta.copy())

    if checkpoint and os.path.exists(CHECKPOINT_FILE):
        Theta = np.load(CHECKPOINT_FILE)

    shape = Theta.shape

    # call optimization method
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    result = minimize(
        fun=cost,
        x0=Theta,
        args=(X, Y, learning_rate),
        # method='TNC',
        # method='SLSQP',
        method='L-BFGS-B',
        # method='BFGS',  # L-BFGS-B
        jac=grad,
        callback=retail,
    )

    Theta = result.x
    Theta.shape = shape
    # Train accuracy
    p = predict_multi_class(Theta, H)
    p += 1   # y is 1-based index in the examples!
    print "Train Accuracy: %f" % (p == y).mean()

def multiclass_with_FNN(checkpoint=True):
    "Similar to Logistic Regression, but using FNN to solve the problem"

    # the code is pretty similar
    data = loadmat('ex3data1.mat')

    # X: 5000 samples x 400 pixels per image (aka features)
    # y: 5000 class belonging vector from 1 to 10 (1-index based!)
    # so in this example, K=10
    X, y = data['X'], data['y']
    Y, mapping = expand_labels(y)

    samples, features = X.shape
    n_klasses = len(mapping)

    learning_rate = 0.1

    # save theta evolution
    trajectory = []

    def retail(theta):
        "save each step of the training"
        # Theta = rebuilt(X, theta)
        # p = predict_multi_class(Theta, X)
        # p += 1   # y is 1-based index in the examples!
        # print "Train Accuracy: %f" % (p == y).mean()
        trajectory.append(theta.copy())

    if checkpoint and os.path.exists(CHECKPOINT_FILE):
        Theta = np.load(CHECKPOINT_FILE)

    # create a FNN that match the number of featurres
    # and 2 hidden layers
    nn = FNN((features, n_klasses))

    nn.solve(X, y, checkpoint='multiclass_with_FNN')


if __name__ == '__main__':
    # multiclass_logistic_regression_single_problem()
    # multiclass_logistic_regression_k_problems()
    multiclass_with_FNN()
