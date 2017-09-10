#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.interpolate

from aiml import *

def main():
    # Load data from CSV
    data = np.genfromtxt('ex1data1.txt', delimiter=',')
    x, y = data[:, 0], data[:, 1]

    # just simply show the data
    plt.plot(x, y, 'rx')
    plt.ylabel('Profit in $10.000s')
    plt.xlabel('Population of City in 10,000s')

    # setup linear regression feature vector with bias column
    X = np.ones_like(data)
    X[:, 1] = x

    # initai seed for gradient descedent
    theta = np.random.random(data.shape[1])
    theta, evolution = gradient_descent(theta, X, y, evolution=True)

    # draw the solution hypotesis
    x_ = np.linspace(np.min(x), np.max(x), 100)
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

    im = plt.imshow(Z, interpolation='bilinear', origin='lower',
                cmap=cm.coolwarm, alpha=0.1,
                extent = (t0.min(), t0.max(), t1.min(), t1.max())
                )

    CS = plt.contour(t0, t1, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Error function')

    ax.plot(theta[0], theta[1], 'rx')  # the solution
    plt.show()

    # 3D plot wit contour
    fig = plt.figure(figsize=(10,5))
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(t0, t1, Z, cmap=cm.coolwarm, alpha=0.3)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    CS = plt.contour(t0, t1, Z)

    ax.view_init(elev=35, azim=-45)  # Select a good point for the camera
    plt.show()

if __name__ == '__main__':
    main()
    print("-End-")
