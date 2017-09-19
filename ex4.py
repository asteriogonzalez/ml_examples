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

    th1 = data['initial_Theta1'].T
    th2 = data['initial_Theta2'].T
    grad = data['grad']
    dth1 = grad[:th1.size].reshape(th1.shape).T
    dth2 = grad[th1.size:].reshape(th2.shape).T

    X = data['X']
    y = data['y']
    lam = data['lambda']
    cost = data['cost']
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
    H = nn.H[0]
    nn.H = nn._forward(H)
    J = nn._cost(lam)
    nn._backprop(alpha)




    foo = 1

if __name__ == '__main__':
    check_bp()
    # test_FFN()
    # multiclass_with_FNN(False)
