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


def H(theta, X):
    "Linear regression Hypotesis"
    return np.dot(X, theta)


def J(theta, X, y):
    "The cost function"
    d = H(theta, X) - y
    return np.dot(d, d) / (2 * len(y))


def simple_lineal_regression():
    "Make a simple linear regression from from Andre Ng Exercise 1"
    # Load data from CSV
    x, y = load_XY_csv('ex1data1.txt')

    # just simply show the data
    plt.plot(x, y, 'rx')
    plt.ylabel('Profit in $10.000s')
    plt.xlabel('Population of City in 10,000s')

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

    # compute the Z error cost surface in a regular mesh
    t0 = np.linspace(theta[0] - 1, theta[0] + 1, 100)
    t1 = np.linspace(theta[1] - 1, theta[1] + 1, 100)

    @np.vectorize
    def cost(*theta):
        return J(theta, X, y)

    t0, t1 = np.meshgrid(t0, t1)
    Z = cost(t0, t1)

    # 2D plot with contours
    fig, ax = plt.subplots(1, 1)

    plt.imshow(Z, interpolation='bilinear', origin='lower',
               cmap=cm.coolwarm, alpha=0.1,
               extent=(t0.min(), t0.max(), t1.min(), t1.max()))

    cs = plt.contour(t0, t1, Z)
    plt.clabel(cs, inline=1, fontsize=10)
    plt.title('Error function')

    ax.plot(theta[0], theta[1], 'rx')  # the solution
    plt.show()

    # 3D plot wit contour
    fig = plt.figure(figsize=(10, 5))
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(t0, t1, Z, cmap=cm.coolwarm, alpha=0.3)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    cs = plt.contour(t0, t1, Z)

    ax.view_init(elev=35, azim=-45)  # Select a good point for the camera
    plt.show()


def multivariate_linear_regression():
    "Linear Regression with regularization and multiple variables"

    # Load data from CSV
    x, y = load_XY_csv('ex1data2.txt')
    plot_mesh(x[:, 0], x[:, 1], y)

    # normalize and keep the normalization info
    x, norm = feature_normalization(x)

    # plot after normalization, to check that the
    # working ranges (both) are O(1)
    # plot_mesh(x[:, 0], x[:, 1], y)

    # initial seed for gradient descent
    X, theta = setup_working_tensors(x)
    theta, _ = gradient_descent(theta, X, y)

    # predict a house price
    sample = (3000, 3)
    price = predict(sample, theta, norm)

    print "Predict price for %d m2 and %d rooms house is $%d" % \
          (sample[0], sample[1], price)


if __name__ == '__main__':
    simple_lineal_regression()
    multivariate_linear_regression()
