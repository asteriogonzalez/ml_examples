#!/usr/bin/env python
"""This file contains the Linear Regression Exercise 1 from
Andrew Ng Coursera course
https://www.coursera.org/learn/machine-learning/home/welcome
buy using Python instead Octave.
"""
from os import path
import random
import time
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt
from scipy.optimize import minimize
from scipy.io import loadmat, savemat
from sklearn.datasets import fetch_mldata

from aiml import *


def test_gradients():
    "Check that numerical and backpropagation get same results"
    sizes = (20 * 20, 10 * 10, 5 * 5, 10)
    # sizes = (20 * 20, 5 * 5, 10)
    nn = FNN(sizes=sizes)

    data = loadmat('ex4data1.mat')
    X, y = data['X'], data['y']
    Y, mapping = expand_labels(y)

    # compute gradients using finite difference and backprop
    lamb = 2
    H = nn.forward(X)
    t0 = time.time()
    grad_num = nn.grad_numerical(X, Y, lamb)
    t1 = time.time()
    grad_bp = nn.grad(X, Y, lamb)
    t2 = time.time()

    diff = grad_num - grad_bp
    for i, th in enumerate(diff):
        error = th.mean()
        assert error < 1e-4
        print "Matrix: %s, error: %s" % (i, error)

    e1 = t1 - t0
    e2 = t2 - t1
    print "Finite Difference: %ds" % e1
    print "Back Propagation:  %ds" % e2
    print "BP/FD: %d faster" % (e1 / e2)


def test_FFN():
    """Load a pre-trained NN from Andrew Ng course and make predictions using
    FNN, then forget the learn, re-train from scratch until same accuracy
    and make the same predictions.

    Finally point out that two NN with same accuracy may have very different
    internal weigths.
    """
    nn = FNN()
    # load a trained NN from disk
    data = loadmat('ex4weights.mat')
    Theta = [data['Theta1'], data['Theta2']]

    for i, Z in enumerate(Theta):
        Z = Z.T
        Theta[i] = Z

    nn.setup(Theta)

    # load sampe labelled data
    data = loadmat('ex4data1.mat')
    X, y = data['X'], data['y']
    Y, mapping = expand_labels(y)

    # evaluate error and accuracy in pre-trained NN
    H = nn.forward(X)
    error =  ((H - Y) ** 2).mean()
    p = nn.predict(X)
    acc = (p == y).mean()
    print "Accurracy: %s, Error:%s (pre-trained)" % (acc, error)

    # solve the problem from scratch
    for i, th in enumerate(Theta):
        Theta[i] = np.random.randn(*th.shape)
    nn.setup(Theta)
    nn.solve(X, y, min_accuracy=0.975, checkpoint='test_FNN.npz')

    # evaluate error and accuracy in our net
    H = nn.forward(X)
    error =  ((H - Y) ** 2).mean()
    p = nn.predict(X)
    acc = (p == y).mean()
    print "Accurracy: %s, Error:%s (this training)" % (acc, error)

    # check the differences between matrices
    for i, th in enumerate(Theta):
        diff = th - nn.Theta[i]
        error = np.abs(diff).mean()
        print "Mean error in Theta[%d]: %f" % (i, error)

    print "Note how 2 NN with same accuracy may have totally different values!"


def test_speed_sigmoid():
    """Check the speed between compute sigmoid again for derivate or
    store H and clone and manipulating the bias column when needed.

    The results show that sigmoid evaluaition is 20-30 times slower
    """
    relspeed = []
    for s in range(1, 200):
        size = (50 * s, 4 * s)
        # size = (50 * s, 4 * s)
        print "Size: %s" % (size, )
        Z = np.random.randn(*size)
        N = 120
        i = N
        t0 = time.time()
        while i > 0:
            S = sigmoid(Z)
            S = S * (1 - S)
            i -= 1
        t1 = time.time()
        e1 = t1 - t0
        print "Sigmoid:\t%f" % e1

        i = N
        t1 = time.time()
        while i > 0:
            S = np.copy(Z)
            S[:, 0] = 1
            i -= 1
        t2 = time.time()
        e2 = t2 - t1

        print "Copy:\t%f" % e2
        rs = e1 / e2
        print "Relative Speed: e1/e2: %f" % rs

        relspeed.append(rs)
        del Z

    plt.plot(relspeed)
    plt.title('Relative speed from copy to sigmoid evaluation')
    plt.show()
    foo = 1


def test_train_nn_with_regular_min_methods():
    chp = Checkpoint('test_train_nn_with_regular_min_methods')
    sizes = (20 * 20, 10 * 10, 5 * 5, 10)
    nn = FNN(sizes=sizes)

    data = loadmat('ex4data1.mat')
    X, y = data['X'], data['y']
    Y, mapping = expand_labels(y)

    lamb = 2
    # nn.solve(X, y, lamb=lamb, checkpoint='test.npz')
    nn.train(X, y, learning_rate=lamb,
                     method='L-BFGS-B',
                     checkpoint=chp)


def test_MNIST_dataset():
    mnist = fetch_mldata('MNIST original')
    print mnist.data.shape
    print mnist.target.shape
    klasses = np.unique(mnist.target)
    n_klasses = len(klasses)

    chp = Checkpoint('test_MNIST_dataset')

    layers = (28 * 28, 14 * 14, 7 * 7, 10)
    nn = FNN(layers)
    for X, y in batch_iter(5000, mnist.data, mnist.target):
        print X.shape, y.shape
        Y, mapping = expand_labels(y, n_klasses)
        break

    del mnist

    # nn.solve(X, Y)
    # foo = 1
    lamb = 2
    # nn.solve(X, y, lamb=lamb, checkpoint='test.npz')
    nn.train(X, Y, learning_rate=lamb,
                     method='L-BFGS-B',
                     checkpoint=chp)




if __name__ == '__main__':
    # test_FFN()
    # test_gradients()
    # test_speed_sigmoid()
    # test_train_nn_with_regular_min_methods()
    test_MNIST_dataset()

    pass
