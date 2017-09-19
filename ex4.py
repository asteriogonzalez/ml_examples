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

from aiml import *


FILE_SEED = 'cache_%s.npz' % path.splitext(path.basename(__file__))[0]
cache = dict()
def safe_step(key, function, *args, **kw):
    global cache
    if not cache and path.exists(FILE_SEED):
        npz = np.load(FILE_SEED)
        cache = npz['cache'].item()

    if key in cache:
        return cache[key]
    t0 = time.time()
    val = function(*args, **kw)
    elapsed = time.time() - t0
    print ">> [%s]: %f secs" % (key, elapsed)
    cache[key] = val
    np.savez_compressed(FILE_SEED, cache=cache )

    print 'saved %s' % key
    return val

def test_FFN(load_data=True):
    # the code is pretty similar
    data = loadmat('ex4weights.mat')
    Theta = [data['Theta1'], data['Theta2']]

    for i, Z in enumerate(Theta):
        Z = Z.T
        # Z = np.insert(X, 0, values=1, axis=len(Z.shape) > 1)  # axis 0 or 1
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



    # X = X[0]
    # Y = Y[0]
    # layers = [t.shape[0] - 1 for t in Theta]
    # layers.append(10)
    # nn.create_netwotk(*layers)

    # Alter Theta, so is not optimized
    for i, th in enumerate(Theta):
        # Theta[i] = np.random.permutation(th)
        Theta[i] = np.random.randn(*th.shape)


    H, YY, Theta = nn.setup(X, Y, Theta)
    H = nn.forward(X)
    error =  ((H - Y) ** 2).mean()
    print error

    grad_numerical = safe_step('grad_numerical', nn._gradients, X, Y)
    # grad_back_prop = nn._BP_gradients()
    grad_back_prop = safe_step('grad_back_prop', nn._BP_gradients)


    g0 = grad_numerical
    g1 = grad_back_prop

    error = g1 - g0

    for i, e in enumerate(error):
        print 'Grad: %d\t%s\tdiff error: %f' % (i, e.shape, e[i].mean())


    # Optimize the NN from scratch
    for i, th in enumerate(Theta):
        Theta[i] = np.random.randn(*th.shape)


    nn.solve(X, y)


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

def check_bp():
    filename = 'bp_example_1.mat'
    data = loadmat(filename)

    th1 = data['Theta1']
    th2 = data['Theta2']
    grad = data['grad']
    dth1 = grad[:th1.size].reshape(th1.shape)
    dth2 = grad[th1.size:].reshape(th2.shape)

    th1 = th1.T
    th2 = th2.T
    dth1 = dth1.T
    dth2 = dth2.T


    X = data['X']
    y = data['y'].astype(np.int8)
    lam = data['lambda']
    cost = data['cost']
    z1 = data['z1']
    z2 = data['z2']
    h1 = data['h1']
    h2 = data['h2']

    first_random = data['first_random']

    samples, features = X.shape

    # setup optimization params
    klasses = np.unique(y)
    n_klasses = klasses.size
    y.shape = (samples, )

    # y[y==10] = 0  # translate label, I don't like 1-based indexes :)
    # setup the Y multiclass matrix

    Y = np.zeros((samples, n_klasses), dtype=np.int8)
    for i, klass in enumerate(y):
        klass = int(klass)
        Y[i][klass - 1] = 1

    Theta = [th1, th2]
    nn = FNN()
    nn.setup(X, Y, Theta)
    H = nn.forward(X)
    J = nn._cost(lam)

    assert np.mean(nn.Zs[1][:, 1:] - z1) < 1e-8
    assert np.mean(nn.Zs[2][:, 1:] - z2) < 1e-8
    assert np.abs(J-cost) < 1e-8

    nn._backprop()

    nn.solve(X, y)


    foo = 1

def test_speed_sigmoid():
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








if __name__ == '__main__':
    # test_speed_sigmoid()
    check_bp()
    # test_FFN()
    # multiclass_with_FNN(False)
