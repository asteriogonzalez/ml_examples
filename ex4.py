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

FILE_SEED = 'Theta.npy'

def test_FFN():
    # the code is pretty similar
    data = loadmat('ex4weights.mat')
    Theta = [data['Theta1'], data['Theta2']]

    for i, Z in enumerate(Theta):
        Z = Z.T
        # Z = np.insert(Z, 0, values=np.ones(Z.shape[0]), axis=1)
        Theta[i] = Z

    nn = FNN()
    nn.Theta = Theta

    data = loadmat('ex4data1.mat')
    X, y = data['X'], data['y']

    samples, features = X.shape

    # setup optimization params
    klasses = np.unique(y)
    n_klasses = klasses.size
    y.shape = (samples, )

    # y[y==10] = 0  # translate label, I don't like 1-based indexes :)
    # setup the Y multiclass matrix
    Y = np.zeros((samples, n_klasses), dtype=np.int8)
    for i, klass in enumerate(y):
        Y[i][klass - 1] = 1


    nn.cost(X, Y)
    foo = 1







def multiclass_with_FNN(load_seed=True):
    "Similar to Logistic Regression, but using FNN to solve the problem"

    # the code is pretty similar
    data = loadmat('ex4weights.mat')

    # X: 5000 samples x 400 pixels per image (aka features)
    # y: 5000 class belonging vector from 1 to 10 (1-index based!)
    # so in this example, K=10
    X, y = data['X'], data['y']

    samples, features = X.shape

    # setup optimization params
    klasses = np.unique(y)
    n_klasses = klasses.size
    y.shape = (samples, )

    y[y==10] = 0  # translate label, I don't like 1-based indexes :)

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

    if load_seed and os.path.exists(FILE_SEED):
        Theta = np.load(FILE_SEED)

    # create a FNN that match the number of featurres
    # and 2 hidden layers
    nn = FNN((features, n_klasses))

    nn.solve(X, y)

    foo = 1


if __name__ == '__main__':
    test_FFN()
    multiclass_with_FNN(False)
