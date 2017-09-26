#!/usr/bin/env python
"""This file contains the Linear Regression Exercise 1 from
Andrew Ng Coursera course
https://www.coursera.org/learn/machine-learning/home/welcome
buy using Python instead Octave.
"""
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

import pytest

from aiml import *

MNIST = 'MNIST original'



study = pytest.mark.study
pre = pytest.mark.pre

def test_normal():
    pass

@pre
def test_foo():
    print "Ok :)"

@pre(name='AI')
class TestUserHandling(object):
    def test_login(self):
        pass
    def test_modification(self):
        assert 1
    def test_deletion(self):
        pass


@study(name='AI')
def test_FFN():
    """This test performs some stages:

    1. Load a pre-trained NN from Andrew Ng course and make predictions using
    FNN

    2. then forget the learn, re-train from scratch until same accuracy
    and make the same predictions.

    3. Finally point out that two NN with same accuracy may have very different
    internal weigths.
    """
    nn = FNN()
    # load a trained NN from disk
    data = loadmat('data/ex4weights.mat')
    Theta = [data['Theta1'], data['Theta2']]

    for i, Z in enumerate(Theta):
        Z = Z.T
        Theta[i] = Z

    nn.setup(Theta)

    # load sampe labelled data
    data = loadmat('data/ex4data1.mat')
    X, y = data['X'], data['y']
    Y, mapping = expand_labels(y)

    # evaluate error and accuracy in pre-trained NN
    H = nn.forward(X)
    error = ((H - Y) ** 2).mean()
    p = nn.predict(X)
    acc = (p == y).mean()
    print "Accurracy: %s, Error:%s (pre-trained)" % (acc, error)


    # solve the problem from scratch
    Y, mapping = expand_labels(y)

    for i, th in enumerate(Theta):
        Theta[i] = np.random.randn(*th.shape)
    nn.setup(Theta)

    nn.train(X, Y)

    # evaluate error and accuracy in our net
    H = nn.forward(X)
    error = ((H - Y) ** 2).mean()
    acc = nn.accuracy(X, Y)
    print "Accurracy: %s, Error:%s (this training)" % (acc, error)

    # check the differences between matrices
    for i, th in enumerate(Theta):
        diff = th - nn.Theta[i]
        error = np.abs(diff).mean()
        print "Mean error in Theta[%d]: %f" % (i, error)

    print "Note how 2 NN with same accuracy may have totally different values!"



@study(name='gradients')
def test_gradients_with_real_dataset():
    "Check that numerical and backpropagation get same results"
    chp = Checkpoint()

    # sizes = (20 * 20, 10 * 10, 5 * 5, 10)
    sizes = (20 * 20, 5 * 5, 10)
    nn = FNN(sizes=sizes)

    Theta = chp.set('Theta', nn.Theta)
    nn.setup(Theta)

    data = loadmat('data/ex4data1.mat')
    X, y = data['X'], data['y']
    Y, mapping = expand_labels(y)

    # compute gradients using finite difference and backprop
    lamb = 2
    nn.forward(X)

    t0 = time.time()
    grad_num = chp.set('grad_num', nn.grad_numerical, X, Y, lamb)

    t1 = time.time()
    grad_bp = nn.grad(X, Y, lamb)

    t2 = time.time()
    diff = grad_num - grad_bp
    diff = diff * diff

    for i, th in enumerate(diff):
        error = th.mean()
        assert error < 1e-3
        print "Matrix: %s, error: %s" % (i, error)

    e1 = t1 - t0
    e2 = t2 - t1
    print "Finite Difference: %ds" % e1
    print "Back Propagation:  %ds" % e2
    print "BP/FD: %d faster" % (e1 / e2)

def test_gradients():
    "Check that numerical and backpropagation get same results"

    # sizes = (20 * 20, 10 * 10, 5 * 5, 10)
    samples = 1000
    sizes = (5 * 5, 10)
    nn = FNN(sizes=sizes)

    X = np.random.randn(samples,sizes[0])
    y = np.random.randint(0, sizes[1], (samples, 1))

    Y, mapping = expand_labels(y)

    # compute gradients using finite difference and backprop
    lamb = 2
    nn.forward(X)

    t0 = time.time()
    grad_num = nn.grad_numerical(X, Y, lamb)

    t1 = time.time()
    grad_bp = nn.grad(X, Y, lamb)

    t2 = time.time()
    diff = grad_num - grad_bp
    diff = diff * diff

    for i, th in enumerate(diff):
        error = th.mean()
        assert error < 1e-3
        print "Matrix: %s, error: %s" % (i, error)

    e1 = t1 - t0
    e2 = t2 - t1
    print "Finite Difference: %ds" % e1
    print "Back Propagation:  %ds" % e2
    print "BP/FD: %d faster" % (e1 / e2)


@study(name='sigmoid')
def test_speed_sigmoid():
    """Study the speed between compute sigmoid again for derivate or
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


@study
def test_MNIST_dataset_with_multimethods():
    """Test some methods or combination of them to train FNN for MNIST
    e.g.
       - GC
       - L-BFGS-B
       - GC for 5 iterations, L-BFGS-B for 5 iterations and so forth
       - GC for 5 iterations, then switch to L-BFGS-B until end
    """

    chp = NullCheckpoint('test_MNIST_dataset_with_multimethods')

    # layers = (28 * 28, 14 * 14, 7 * 7, 10)
    layers = (28 * 28, 10 * 10, 10)
    results = dict()
    methods = list()
    methods.extend(['CG', 'L-BFGS-B', ])
    # methods.extend([[('CG', 5), ('L-BFGS-B', 5)]])
    methods.extend([[('CG', 5), ('L-BFGS-B', 10 ** 4)]])

    n_klasses = batch_mldata_classes(MNIST)
    for i, (X, y) in batch_mldata(MNIST, 5000):
        print "Batch [%3d] %s, %s" % (i, X.shape, y.shape)
        Y, mapping = expand_labels(y, n_klasses)
        break

    for method in methods:
        mhash = get_hashable(method)
        agent = FNN(layers)
        lamb = 2
        s = "Training %s with %s method" % (agent.__class__.__name__, mhash)
        print
        print s
        print "-" * len(s)

        t0 = time.time()
        results[mhash] = r = agent.train(
            X, Y, learning_rate=lamb,
            method=method,
            checkpoint=chp,
            maxiter=200)
        r.elapsed = time.time() - t0
        print "Method: %s : %d secs" % (method, r.elapsed)

    pplot(results, 'error', 'accuracy')


@study
def test_MNIST_training():
    """Train a FNN for MNIST using batch
    """
    chp = Checkpoint()

    layers = (28 * 28, 14 * 14, 7 * 7, 10)
    # layers = (28 * 28, 10 * 10, 10)

    # method = [('CG', 5), ('L-BFGS-B', 10 ** 4)]
    method = 'L-BFGS-B'

    agent = FNN(layers)
    lamb = 2
    s = "Training %s with %s method" % (agent.__class__.__name__, method)
    print
    print s
    print "-" * len(s)

    n_klasses = batch_mldata_classes(MNIST)
    batch_size = 500
    batch = batch_mldata(MNIST, batch_size)
    for i, (X0, y0) in batch:
        print "Batch [%3d] %s, %s" % (i, X0.shape, y0.shape)
        Y0, mapping = expand_labels(y0, n_klasses)
        break  # initial batch

    accuracy = list()

    for i, (X, y) in batch:
        t0 = time.time()
        r = agent.train(
            X0, Y0, learning_rate=lamb,
            method=method,
            checkpoint=chp,
            maxiter=int(batch_size / 7.5),
        )
        r.elapsed = time.time() - t0

        # check accuracy with UNSEEN data
        # and use for learning in the next step

        Y, mapping = expand_labels(y, n_klasses)
        agent.forward(X)  # accuracy expects a previous FWD pass
        acc = agent.accuracy(X, Y)
        print "Batch [%3d] accuracy: %f,  %d secs" % (i, acc, r.elapsed)
        accuracy.append(acc)
        X0, Y0 = X, Y
        if i > 30:
            break

    results = dict(accuracy=accuracy)
    pplot(results, 'accuracy')


if __name__ == '__main__':
    pass
    test_gradients()
    # test_FFN()
    # study_speed_sigmoid()
    # study_MNIST_dataset_with_multimethods()
    # study_MNIST_training()
