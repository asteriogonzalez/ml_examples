#!/usr/bin/env python
"""This file contains the Linear Regression Exercise 1 from
Andrew Ng Coursera course
https://www.coursera.org/learn/machine-learning/home/welcome
buy using Python instead Octave.
"""
import sys
from os import path
import random
import time
import zlib
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
    chp = Checkpoint()
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


MNIST = 'MNIST original'
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
    foo = 1


def test_MNIST_training():
    """Train a FNN for MNIST using batch
    """
    chp = Checkpoint()

    layers = (28 * 28, 14 * 14, 7 * 7, 10)
    # layers = (28 * 28, 10 * 10, 10)

    method = [('CG', 5), ('L-BFGS-B', 10 ** 4)]
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
    foo = 1


dbname = 'example.db'
def test_save_sqlite_arrays():
    "Load MNIST database (70000 samples) and store in a compressed SQLite db"
    os.path.exists(dbname) and os.unlink(dbname)
    con = sqlite3.connect(dbname, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    cur.execute("create table test (idx integer primary key, X array, y integer );")

    mnist = fetch_mldata('MNIST original')

    X, y =  mnist.data, mnist.target
    m = X.shape[0]
    t0 = time.time()
    for i, x in enumerate(X):
        cur.execute("insert into test (idx, X, y) values (?,?,?)",
                    (i, y, int(y[i])))
        if not i % 100 and i > 0:
            elapsed = time.time() - t0
            remain = float(m - i) / i * elapsed
            print "\r[%5d]: %3d%% remain: %d secs" % (i, 100 * i / m, remain),
            sys.stdout.flush()

    con.commit()
    con.close()
    elapsed = time.time() - t0
    print
    print "Storing %d images in %0.1f secs" % (m, elapsed)

def test_load_sqlite_arrays():
    "Query MNIST SQLite database and load some samples"
    con = sqlite3.connect(dbname, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()

    # select all images labeled as '2'
    t0 = time.time()
    cur.execute('select idx, X, y from test where y = 2')
    data = cur.fetchall()
    elapsed = time.time() - t0
    print "Retrieve %d images in %0.1f secs" % (len(data), elapsed)







def test_checkpoint():

    chp = Checkpoint()
    chp['foo'] = 'bar'
    chp.save()

    filename = chp.filename
    del chp

    chp2 = Checkpoint(filename)
    assert chp2['foo'] == 'bar'



if __name__ == '__main__':
    test_checkpoint()
    # test_FFN()
    # test_gradients()
    # test_speed_sigmoid()
    # test_train_nn_with_regular_min_methods()
    # test_MNIST_dataset_with_multimethods()
    test_MNIST_training()
    # test_save_sqlite_arrays()
    # test_load_sqlite_arrays()

    pass
