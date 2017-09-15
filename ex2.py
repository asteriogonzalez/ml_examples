#!/usr/bin/env python
"""This file contains the Linear Regression Exercise 1 from
Andrew Ng Coursera course
https://www.coursera.org/learn/machine-learning/home/welcome
buy using Python instead Octave.
"""
import math
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt

from aiml import *


def plot_sigmoid():
    "Plot the sigmoid function"
    x = np.linspace(-8, 8, 100)
    y = sigmoid(x)
    plt.plot(x, y, 'b-')
    plt.show()


def cost(theta, X, y, lam=0):
    "Compute the cost of logistic regression"
    h = sigmoid(np.dot(X, theta))  # 0 < h < 1
    cost = -y * np.log(h) - (1 - y) * np.log(1 - h)
    cost = cost.sum()
    if lam > 0:  # regularization ignore bias parameter
        theta2 = np.copy(theta)
        theta2[0] = 0
        reg = lam * np.dot(theta2, theta2) / 2.0
        cost += reg

    cost /= y.size

    return cost


def grad(theta, X, y, lam=0):
    "Compute the gradient of logistic regression"
    h = sigmoid(np.dot(X, theta))
    d = h - y
    grad = np.dot(d, X) / y.size

    if lam > 0:  # regularization ignore bias parameter
        # theta2 = np.zeros_like(theta)
        # theta2[1:] = theta[1:]

        theta2 = np.copy(theta)
        theta2[0] = 0

        reg = lam * theta2 / y.size
        grad += reg

    return grad


def simple_logistic_regression():
    "Make a simple linear regression from from Andre Ng Exercise 1"
    # Load data from CSV
    x, y = load_XY_csv('ex2data1.txt')
    x0 = x[y <= 0]
    x1 = x[y > 0]

    # just simply show the data
    plt.plot(x0[:, 0], x0[:, 1], 'yo')
    plt.plot(x1[:, 0], x1[:, 1], 'k+')

    plt.ylabel('Exam 2 score')
    plt.xlabel('Exam 1 score')
    # plt.show()

    # find the solution
    X = map_feature(x[:, 0], x[:, 1])
    theta = np.random.random(X.shape[1]) / 1000

    # save theta evolution
    trajectory = []

    def retail(x):
        "hello"
        trajectory.append(x.copy())

    # call optimization method
    result = opt.fmin_l_bfgs_b(
        cost,
        x0=theta,
        fprime=grad,
        args=(X, y, ),
        disp=True,
        epsilon=0.001,
        callback=retail,
    )
    sol = result[0]

    # plot the decision boundary
    # we have 2 alternatives for plotting decision boundaries
    # 1. as the solution is a line, we can find 2 points
    # the line where hypothesis(theta) = 0
    # if x1 = 0 --> x2 = -t0 / t2
    # if x2 = 0 --> x1 = -t0 / t1
    # t0, t1, t2 = sol
    # x2 = -t0 / t1
    # x1 = -t0 / t2
    # plt.plot([0, x1], [x2, 0], 'b-')
    # plt.show()

    # 2. Use a more general approach when the shape of the
    # boundary is unknown (maybe we increase the number
    # of features)

    u = np.linspace(10, 150, 50)
    v = np.linspace(10, 150, 50)

    @np.vectorize
    def hypothesis(*theta):
        X = map_feature(*theta)
        return X.dot(sol)

    plot_decision_boundary(hypothesis, u, v)

    # plot error evolution
    plot_error_evolution(trajectory, cost, X, y)

    # Train accuracy
    p = predict(sol, X)
    print "Train Accuracy: %f" % (p == y).mean()


def regularized_logistic_regression(lam=1, degree=3):
    "Make a simple linear regression from from Andre Ng Exercise 1"
    # Load data from CSV
    x, y = load_XY_csv('ex2data2.txt')
    x0 = x[y <= 0]
    x1 = x[y > 0]

    # just simply show the data
    plt.plot(x0[:, 0], x0[:, 1], 'yo')
    plt.plot(x1[:, 0], x1[:, 1], 'k+')

    plt.ylabel('Microship Test 2')
    plt.xlabel('Microship Test 1')
    # plt.show()

    # find the solution
    X = map_feature(x[:, 0], x[:, 1], degree=degree)
    theta = np.random.random(X.shape[1]) / 1000
    # theta = np.ones_like(theta)

    # save theta evolution for plotting error curve
    trajectory = []

    def retail(x):
        trajectory.append(x.copy())

    # call optimization method
    result = opt.fmin_l_bfgs_b(
        cost,
        x0=theta,
        fprime=grad,
        args=(X, y, lam),
        disp=False,
        epsilon=0.001,
        callback=retail,
    )
    sol = result[0]

    # plot the decision boundary
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)

    @np.vectorize
    def hypothesis(*theta):
        X = map_feature(*theta, degree=degree)
        return X.dot(sol)

    plot_decision_boundary(hypothesis, u, v)

    # plot error evolution
    plot_error_evolution(trajectory, cost, X, y, lam)

    # Train accuracy
    p = predict(sol, X)
    print "Train Accuracy: %f" % (p == y).mean()


if __name__ == '__main__':
    plot_sigmoid()
    simple_logistic_regression()
    regularized_logistic_regression()

    # check for params variations
    regularized_logistic_regression(lam=0.001, degree=6)
    regularized_logistic_regression(lam=0.001, degree=9)
    regularized_logistic_regression(lam=0.1, degree=10)
    regularized_logistic_regression(lam=1, degree=2)

    # high over-optimization
    regularized_logistic_regression(lam=0.0, degree=7)
