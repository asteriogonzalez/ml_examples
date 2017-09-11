#!/usr/bin/env python
"""This file contains the Linear Regression Exercise 1 from
Andrew Ng Coursera course
https://www.coursera.org/learn/machine-learning/home/welcome
buy using Python instead Octave.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt

from aiml import gradient_descent, feature_normalization, \
    load_XY_csv, plot_mesh, setup_working_tensors, predict, \
    sigmoid


def plot_sigmoid():
    "Plot the sigmoid function"
    x = np.linspace(-8, 8, 100)
    y = sigmoid(x)
    plt.plot(x, y, 'b-')
    plt.show()


def cost(theta, X, y):
    "Compite the cost of logistic regression"
    z = np.dot(X, theta)
    h = sigmoid(z)  # 0 < h < 1
    cost = -y * np.log(h) - (1 - y) * np.log(1 - h)
    return cost.mean()


def grad(theta, X, y):
    "Compute the gradient of logistic regression"
    h = sigmoid(np.dot(X, theta))
    d = h - y
    grad = np.dot(d, X) / y.size
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
    X, theta = setup_working_tensors(x)
    theta = theta / 1000

    result = opt.fmin_bfgs(
        f=cost,
        x0=theta,
        fprime=grad,
        args=(X, y),
        disp=True,
        retall=True,
        full_output=True,
    )
    print result

    # plot the decision boundary
    # (is a line in this case)
    # find points where h(theta) > 0.5
    # h = sigmoid(np.dot(X, theta))
    # X . theta == 0 ?
    # t0 + x1*t1 + x2*t2 == 0?
    # take two known points
    # if x1 = 0 --> x2 = -t0 / t2
    # if x2 = 0 --> x1 = -t0 / t1

    t0, t1, t2 = result[0]  # theta
    x2 = -t0 / t1
    x1 = -t0 / t2

    plt.plot([0, x1], [x2, 0], 'b-')
    plt.show()

    # plot error evolution
    trajectory = result[7]  # all steps
    error = [cost(t, X, y) for t in trajectory]
    plt.plot(error)
    plt.show()


if __name__ == '__main__':
    # plot_sigmoid()
    simple_logistic_regression()
