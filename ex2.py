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
# import scipy.interpolate

from aiml import gradient_descent, feature_normalization, \
    load_XY_csv, plot_mesh, setup_working_tensors, predict


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

    plt.show()

    # initial seed for gradient descent
    X, theta = setup_working_tensors(x)
    theta, evolution = gradient_descent(theta, X, y, evolution=True)

    # draw the solution hypothesis
    x_ = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
    y_ = theta[0] + theta[1] * x_
    plt.plot(x_, y_, 'b-')
    plt.show()

    # show the error function across iterations
    plt.plot(evolution)
    plt.show()


if __name__ == '__main__':
    simple_logistic_regression()
