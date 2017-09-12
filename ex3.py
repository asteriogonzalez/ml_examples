#!/usr/bin/env python
"""This file contains the Linear Regression Exercise 1 from
Andrew Ng Coursera course
https://www.coursera.org/learn/machine-learning/home/welcome
buy using Python instead Octave.
"""
import os
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


def rebuilt(X, theta):
    """Reshape the Theta tensor in single vector suitable for
    optimization algorithms"""
    features = X.shape[1]
    klasses = theta.size // features

    assert features * klasses == theta.size  # perfect fitting

    Theta = np.reshape(theta, (features, klasses))
    return Theta


def cost(theta, X, Y, lam=0):
    "Compute the cost of logistic regression"
    n = Y.shape[0]
    Theta = rebuilt(X, theta)
    h = sigmoid(np.dot(X, Theta))

    cost = -Y * np.log(h) - (1 - Y) * np.log(1 - h)
    cost = cost.sum()
    if lam > 0:  # regularization ignore bias parameter
        theta2 = np.copy(theta)
        theta2[0] = 0
        reg = lam * np.dot(theta2, theta2) / 2.0
        cost += reg

    cost /= n

    print "COST:", cost
    return cost


def grad(theta, X, Y, lam=0):
    "Compute the gradient of logistic regression."
    n = Y.shape[0]

    Theta = rebuilt(X, theta)
    H = sigmoid(np.dot(X, Theta))

    d = H - Y
    grad = X.T.dot(d)

    if lam > 0:  # regularization ignore bias parameter
        # theta2 = np.zeros_like(theta)
        # theta2[1:] = theta[1:]
        Theta2 = np.copy(Theta)
        Theta2[0] = 0
        reg = lam * Theta2
        grad += reg

    grad /= n

    # restore same shape that minimization algorithms uses
    grad.shape = theta.shape
    return grad


def multiclass_logistic_regression():
    "Make a simple linear regression from from Andre Ng Exercise 1"
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
    X = np.insert(X, 0, values=np.ones(samples), axis=1)
    # setup the Y multiclass matrix
    Y = np.zeros((samples, n_klasses), dtype=np.int8)
    for i, klass in enumerate(y):
        Y[i][klass - 1] = 1

    learning_rate = 0.1
    # save theta evolution
    trajectory = []

    def retail(x):
        "save each step of the training"
        print x
        trajectory.append(x.copy())

    # call optimization method
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    result = minimize(
        fun=cost,
        x0=Theta,
        args=(X, Y, learning_rate),
        # method='TNC',
        method='BFGS',  # L-BFGS-B
        jac=grad,
        callback=retail,
    )

    # Train accuracy
    p = predict_multi_class(Theta, X)
    p += 1   # y is 1-based index in the examples!
    print "Train Accuracy: %f" % (p == y).mean()


def multiclass_logistic_regression_split():
    "Make a simple linear regression from from Andre Ng Exercise 1"
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
    X = np.insert(X, 0, values=np.ones(samples), axis=1)
    y.shape = (samples, )

    # we solve (k) simple problems, using different
    # Theta columns and building a custom y vector for
    # each problem
    learning_rate = 0.1

    # check if we have solved the problem before, just for fast debugging
    path = 'Theta.npy'
    if os.path.exists(path):
        Theta = np.load(path)
    else:
        for k, klass in enumerate(klasses):
            y_i = (y == (k + 1)) * 1
            theta = Theta[:, k]
            theta, result = solve_logistic_regression(
                X, y_i, theta, learning_rate)
            Theta[:, k] = theta
            print "Class %s : error: %s, %s" % (k, result.fun, result.message)
        np.save(path, Theta)

    # Train accuracy
    p = predict_multi_class(Theta, X)
    p += 1   # y is 1-based index in the examples!
    print "Train Accuracy: %f" % (p == y).mean()


if __name__ == '__main__':
    multiclass_logistic_regression_split()
    # multiclass_logistic_regression()
