"""Some convenient functions for all exercises
"""

import time
import os
import numpy as np
import cPickle as pickle
import random
import itertools
import types
# # import bz2
import gzip
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
from scipy.optimize import minimize

# TODO: test for checking numerical and backprop gradients
# TODO: solve net convergence using L-BFGS-B method (or any other in 1-D form)
# TODO: the idea is to expose the NN problems as a 1-D optimization problem.
# TODO: check the extra column in Theta when Theta is optimized or from scratch

def batch_iter(size=1000, *data):
    m = data[0].shape[0]
    index = range(m)
    random.shuffle(index)
    for i in xrange(0, m, size):
        result = list()
        idx = index[i:i+size]
        for element in data:
            result.append(element[idx])
        yield result


class NullCheckpoint(dict):
    "A null checkpoint that does nothing"
    def __init__(self, filename):
        self.filename = filename

    def save(self, **data):
        "save the dict to disk"
        pass

    def load(self):
        "load dict from disk"
        return self


class Checkpoint(dict):
    """A simple class for saving and retrieve info
    to continue interrupted works
    """
    head = 'chp_'
    # ext = '.pbz2'
    ext = '.pzip'

    # compressor = bz2.BZ2File
    compressor = gzip.GzipFile
    def __init__(self, filename):
        if not filename.startswith(self.head):
            filename = self.head + filename

        if not filename.endswith(self.ext):
            filename += self.ext
        self.filename = filename

    def save(self, **data):
        "save the dict to disk"
        print ">> Saving checkpoint: %s" % self.filename
        for k, v in self.items():
            if k not in data:
                data[k] = v

        with self.compressor(self.filename, 'wb') as f:
            pickle.dump(data, f, protocol=2)

    def load(self):
        "load dict from disk"
        if os.access(self.filename, os.F_OK):
            print "<< Loading checkpoint: %s" % self.filename
            with self.compressor(self.filename, 'rb') as f:
                data = pickle.load(f)
            self.update(data)
        return self


def load_XY_csv(filename):
    """Load a CSV file and prepare the X e y matrix
    with the bias term ready for optimization.
    """
    data = np.genfromtxt(filename, delimiter=',')
    x, y = data[:, :-1], data[:, -1]
    return x, y


def gradient_descent(theta, X, y, alpha=0.01, iters=20000,
                     err=0.00001, evolution=False):
    "Simple Gradient Descent implementation"

    progress = list()
    n = float(len(y))
    n2 = 2 * n
    rate = alpha / n
    last_err = float('inf')
    for i in xrange(iters):
        h = np.dot(X, theta)
        d = h - y
        J = np.dot(d, d) / n2
        s = np.dot(d, X)
        theta -= rate * s
        if not i % 100:
            print '[%4d] err: %0.5f' % (i, J)

        if (last_err - J) < err:
            break
        last_err = J
        if evolution:
            progress.append(last_err)

    return theta, progress


def feature_normalization(x):
    "Normalize each feature and returns the mean and std used for each column"

    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    X = (x - mean) / std

    return X, (mean, std)


def revert_feature(x, norm):
    "revert from normalizer space to real one"
    mean, std = norm
    return (x * std) + mean


def reduce_feature(x, norm):
    "convert to normalize space"
    mean, std = norm
    return (x - mean) / std


def setup_working_tensors(x):
    "Prepare X tensor and guess a initial seed for theta"
    # set X with bias term
    shape = list(x.shape)
    shape[1] += 1

    X = np.ones(shape=shape)
    X[:, 1:] = x

    theta = np.random.random(X.shape[1])

    return X, theta


def plot_mesh(x, y, z):
    "Plot a mesh of non-regular data points"
    x_ext = x.min(), x.max()
    y_ext = y.min(), y.max()
    xi = np.linspace(*x_ext, num=100)
    yi = np.linspace(*y_ext, num=100)
    # grid the data.
    zi = griddata(x, y, z, xi, yi, interp='linear')

    # contour the gridded data, plotting dots at
    # the nonuniform data points.
    plt.contour(xi, yi, zi, linewidths=0.5, colors='k', alpha=0.2)
    plt.contourf(xi, yi, zi,
                 vmax=abs(zi).max(), vmin=-abs(zi).max(),
                 alpha=0.2)
    plt.colorbar()  # draw colorbar
    # plot data points.
    # plt.scatter(x, y, marker='x')  #, s=5, zorder=10)
    plt.plot(x, y, 'rx')  # , s=5, zorder=10)
    plt.xlim(*x_ext)
    plt.ylim(*y_ext)
    plt.show()


def create_grid_around(mean, std):
    "Create a grid around mean and std"
    wide = 1.5 * std

    t0 = np.linspace(mean[0] - wide[0], mean[0] + wide[0], 5)
    t1 = np.linspace(mean[1] - wide[1], mean[1] + wide[1], 5)

    t0, t1 = np.meshgrid(t0, t1)
    return t0, t1


def sigmoid(z):
    "The sigmoid function"
    return 1.0 / (1 + np.exp(-z))


def map_feature(x1, x2, degree=1):
    """Create a full feature vector from two sets.
    """
    assert np.array(x1).shape == np.array(x2).shape

    # just for didactic purposes
    # shape = (x1.size, 1)
    # out = np.empty((x1.size, 0))
    # for i in range(0, degree + 1):
        # for j in range(i + 1):
            # print "(%s, %s)" % ((i - j, j))
            # feature = (x1 ** (i - j)) * (x2 **j)
            # feature.shape = shape
            # # append each new column
            # out = np.append(out, feature, axis=1)

    #  the most resilient way, no matter the data type input
    out = [] # start with the bias term
    for i in range(0, degree + 1):
        for j in range(i + 1):
            feature = (x1 ** (i - j)) * (x2 ** j)
            out.append(feature)

    out = np.array(out).T

    # # slightly faster (save 1 loop and 1st is forced to 1)
    # shape = (x1.size, 1)
    # out = np.ones(shape) # start with the bias term
    # for i in range(1, degree + 1):
        # for j in range(i + 1):
            # # print "(%s, %s)" % ((i - j, j))
            # feature = (x1 ** (i - j)) * (x2 **j)
            # feature.shape = shape
            # # append each new column
            # out = np.append(out, feature, axis=1)

    return out


def predict(theta, X):
    "Predict the probability of a sample X of being a 'true'"
    h = sigmoid(np.dot(X, theta))
    p = (h >= 0.5)
    return p

def predict_multi_class(Theta, X):
    "return the predictions for a sets"
    h = sigmoid(X.dot(Theta))
    p = np.argmax(h, axis=1)
    return p


def plot_error_evolution(trajectory, cost, *args):
    "Plot error evolution, to see the shape"
    # TODO: obsolete plot_error_evolution
    error = [cost(t, *args) for t in trajectory]
    plt.plot(error)
    plt.ylabel('Error')
    plt.xlabel('Iterations')
    plt.show()

def pplot(data, *keys, **options):
    n = len(keys) or 1
    fig, axes = plt.subplots(n, sharex=True)

    if keys:
        for label, values in data.items():
            for i, key in enumerate(keys):
                axes[i].plot(values.get(key), label=label, **options)
        for ax in axes:
            legend = ax.legend(loc='upper center')
    else:
        for label, values in data.items():
            axes.plot(values, label=label, **options)

        legend = axes.legend(loc='upper center')



    plt.show()


def plot_decision_boundary(hypothesis, u, v):
    "Plot the decision boundary where hypothesis is zero"
    u, v = np.meshgrid(u, v)

    Z = hypothesis(u, v)

    # the decision boundary if where hypothesis is zero
    # we can figure out drawing the 'zero' contour line
    plt.contour(u, v, Z, levels=[0], linewidths=2.5,
                colors='g', linestyles='dashed')

    # draw some iso-lines, to see the shape of the solution
    levels = range(-5, 5)
    plt.contour(u, v, Z, levels=levels, linewidths=1.5,
                colors='k', alpha=0.05)
    plt.contourf(u, v, Z, levels=levels, alpha=0.05)

    plt.title('Decision boundary')
    plt.show()

def solve_logistic_regression(X, y, theta=None, learning_rate=1.0):
    """Solve Logistic Regression problems.

    Define some internal functions for cost and gradient inside
    this scope for convenience.

    If theta is passed, it will be used as initial seed, other wise
    a random seed will be generated.

    X: A  matrix with: m sample x (n+1) features matrix.
       The bias column will be expected, otherwise one will be inserted.
    """

    # def rebuilt(X, theta):
        # """Reshape the Theta tensor in single vector suitable for
        # optimization algorithms"""
        # features =  X.shape[1]
        # klasses = theta.size // features

        # assert features * klasses == theta.size  # perfect fitting

        # Theta = np.reshape(theta, (features, klasses))
        # return Theta


    def cost(theta, X, y, learning_rate=0):
        "Compute the cost of logistic regression"
        h = sigmoid(np.dot(X, theta))

        cost = -y * np.log(h) - (1 - y) * np.log(1 - h)
        cost = cost.sum()
        if learning_rate > 0:  # regularization ignore bias parameter
            theta2 = np.copy(theta)
            theta2[0] = 0
            reg = learning_rate * np.dot(theta2, theta2) / 2.0
            cost += reg

        cost /= y.shape[0]

        # print "COST:", cost
        return cost


    def grad(theta, X, y, learning_rate=0):
        "Compute the gradient of logistic regression."
        h = sigmoid(np.dot(X, theta))
        d = h - y
        grad = X.T.dot(d)

        if learning_rate > 0:  # regularization ignore bias parameter
            # theta2 = np.zeros_like(theta)
            # theta2[1:] = theta[1:]
            theta2 = np.copy(theta)
            theta2[0] = 0
            reg = learning_rate * theta2
            grad += reg

        grad /= y.shape[0]

        # restore same shape that minimization algorithms uses
        # grad.shape = theta.shape
        return grad

    # save theta evolution
    trajectory = []

    def retail(x):
        "save each step of the training"
        # print x
        trajectory.append(x.copy())


    if not (X[:, 0] == 1).all():
        raise RuntimeError("""X miss the bias term.
        Please insert using:
        X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)
        """)

    # call optimization method
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    result = minimize(
        fun=cost,
        x0=theta,
        args=(X, y, learning_rate),
        # method='TNC',
        # method='SLSQP',
        method='L-BFGS-B',
        # method='BFGS',
        jac=grad,
        callback=retail,
        # options={'maxiter': 100000, 'disp': False}
    )

    theta = result.x
    return theta, result



# def strip_bias(H):
    # if len(H.shape) > 1:
        # return H[:, 1:]
    # return H[1:]

def num_samples(H):
    if len(H.shape) > 1:
        return H.shape[0]
    return 1


#-------------------------------------
# Optimization support
#-------------------------------------

def pack(theta, matrices):
    "pack all data contained in 1-D array into several matrices"
    a = b = 0
    for th in matrices:
        b += th.size
        th.flat = theta[a:b]   # load data into matrix, but change the shape
        a = b
    assert b == theta.size  # all data has been used

def unpack(matrices):
    "unroll all data contained in several matrices into a single 1-D vector"
    return np.concatenate(
        [th.ravel() for th in matrices]
    )

class Agent(object):
    """A interface for agent optimizers"""
    def __init__(self):
        self.Theta = list()
        self.J = 0

    def setup(self, Theta):
        "Setup NN parameters (Theta weigths by now)"
        self.Theta = np.array(Theta)

    def cost(self, *args, **kw):
        raise RuntimeError('you call an abstract method')

    def grad(self, *args, **kw):
        raise RuntimeError('you call an abstract method')

    def train(self, *args, **kw):
        """Train the agent exposing cost and gradient functions in
        1-D vector form, so we can use advanced methods like L-BFGS-B, CG
        or any other from scipy.minimize module."""

        raise RuntimeError('you call an abstract method')

def expand_labels(y, n_klasses=None):
    """Expand a 1-D label vector into a matrix with a '1'
    in the right column and zeros elsewhere.

    If the y sample hasn't elements of ALL classes (is not fully populated)
    then you must provide n_klasses parameter to build the matrix with
    the right (full) dimension.
    """
    if len(y.shape) == 1:
        klasses = list(np.unique(y))
        klasses.sort()

        n_klasses = n_klasses or len(klasses)

        mapping = dict()
        for i, label in enumerate(klasses):
            mapping[label] = i

        samples = y.shape[0]
        y.shape = (samples, )

        # setup the Y multiclass matrix
        Y = np.zeros((samples, n_klasses), dtype=np.int8)
        for i, label in enumerate(y):
            Y[i][mapping[label]] = 1

        return Y, mapping
    else:
        return y, {}  # we missing the mapping, you know in caller level.



def get_hashable(method):
    if method.__hash__:
        return method
    return str(method)


class ClassificationAgent(Agent):
    def __init__(self):
        Agent.__init__(self)

    def train(self, X, y, learning_rate=1.0, method='L-BFGS-B',
                    plot=False, checkpoint=None,
                    **options):

        """Train the classification agent.

        The agent must be already configured (layers in NN, etc) prior
        calling this function.

        X: the features training data set
        y: the labels data set
        method: minimization method to be used, L-BFGS-B by default.
        learning_rate: the regularization parameter (lambda)
        """
        Y, mapping = expand_labels(y)

        def cost(theta, X, Y, learning_rate=0):
            "Compute the cost of NN"
            self.pack(theta)
            return self.cost(X, Y, lamb=learning_rate)

        def grad(theta, X, Y, learning_rate=0):
            "Compute the gradient of logistic regression."
            # force forward pass
            # cost(theta, X, Y, learning_rate)
            # pack(theta, nn.Theta) (not necessary)

            grads = self.grad(X, Y, lamb=learning_rate)
            return unpack(grads)

        # save theta evolution
        checkpoint.load()
        # error = checkpoint.setdefault('error', list())
        # trajectory = checkpoint.setdefault('trajectory', list())
        # accuracy = checkpoint.setdefault('accuracy', list())

        error = list()
        trajectory = list()
        accuracy = list()

        def retail(x):
            "save each step of the training"
            acc = self.accuracy(X, Y)
            trajectory.append(x.copy())
            error.append(self.J)
            accuracy.append(acc)
            iteration = len(accuracy)
            print "[%4d] Accurracy: %f\tCost: J=%f" % \
                  (iteration, acc, self.J)

            if not len(error) % 20:
                checkpoint.save(Theta=self.Theta, iteration=iteration)

        Theta = checkpoint.get('Theta')
        if Theta is not None:
            self.setup(Theta)

        args=(X, Y, learning_rate)
        # force a 1st time cost computation to assure that
        # gradient will be properly computed (e.g backpropagation case)
        self.cost(*args)

        theta = unpack(self.Theta)
        ops = {'maxiter': 400, 'disp': False, 'gtol': 1e-3,}
        ops.update(options)

        # prepare for cycling methods is they're specified.
        if isinstance(method, (types.ListType, types.TupleType)):
            methods = method
        else:
            methods = [(method, ops['maxiter']), ]

        remain_iters = ops['maxiter']
        for method, iters in itertools.cycle(methods):

            # call optimization method
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
            ops['maxiter'] = iters = min(remain_iters, iters)
            print "Using: %s for %d iters" % (method, iters)

            result = minimize(
                fun=cost,
                x0=theta,
                args=args,
                method=method,
                jac=grad,
                callback=retail,
                options=ops,
            )
            remain_iters -= iters
            if remain_iters <= 0:
                break

            theta = unpack(self.Theta)  # for the next method (is any)

        checkpoint.save(Theta=self.Theta)
        if plot:
            # Plot evolution
            fig, (ax0, ax1,) = plt.subplots(2, sharex=True)
            ax0.plot(errors)
            ax1.plot(accuracy)
            ax0.set_title('Error J')
            ax1.set_title('Accurracy')
            plt.show()

        result.error = error
        result.accuracy = accuracy
        return result


class FNN(ClassificationAgent):
    def __init__(self, sizes=None):
        ClassificationAgent.__init__(self)

        # self.Z = list()  # intermediate Z output values for backprop
        self.H = None    # the last hypothesis value
        self.Hs = list()
        # self.X = None
        # self.Y = None

        if sizes:
            self.create_netwotk(sizes)

    def create_netwotk(self, sizes):
        """Create the network structure.
        sizes contains the number of nodes in each layer.

        size[0] is the number of input features, without bias term
        size[1] is the number of nodes (sub-features) without bias
        ...
        size[n] is the number of nodes at the end, usually match with
                the number of classes in classification problems.
        """
        self.Theta = list()
        for i in range(0, len(sizes) - 1):
            # use extended sizes with bias term for convenience
            # but remove in the last stage
            size = (sizes[i] + 1, sizes[i + 1])
            # create and initialize layer with random values
            # following the advice from Andre Ng
            epsilon = np.sqrt(6) / np.sqrt(sum(size))
            Theta = np.random.randn(*size) * epsilon
            self.Theta.append(Theta)



    def solve(self, X, y, alpha=1.0, lamb=0.1, min_accuracy=0.95, checkpoint='',
              plot=False):
        """Train the Net until some criteria is reached.

        X: the training samples
        y: the """
        self.X = X
        Y, mapping = expand_labels(y)

        chp = Checkpoint(checkpoint)

        cache = chp.load()
        if cache:
            self.setup(cache['Theta'])

        errors = []
        deltas = []
        accuracy = []

        last_acc = 0.0
        last_J = float('inf')
        i0 = cache.get('iteration', 0)
        for iteration in xrange(i0, 10**5):
            J = self.cost(X, Y, lamb)
            self.backprop(X, Y, alpha, lamb)

            if not iteration % 20:
                acc = self.accuracy(X, Y)
                delta = np.mean([d.mean() for d in self.H])

                accuracy.append(acc)
                deltas.append(delta)
                errors.append(J)

                print "[%4d] Accurracy: %s, Cost: J=%s" % \
                              (iteration, acc, J)

                if round(J, 9) >= round(last_J, 9) or \
                   delta < 1e-3 or J < 1e-5 or \
                   acc > min_accuracy:
                    print "Max Accurracy: %f : STOP" % last_acc
                    break

                if checkpoint:
                    chp.save(
                        Theta=self.Theta,
                        iteration=iteration,
                        )
                last_acc, last_J = acc, J

        if plot:
            # Plot evolution
            fig, (ax0, ax1, ax2) = plt.subplots(3, sharex=True)
            ax0.plot(errors)
            ax1.plot(accuracy)
            ax2.plot(deltas)
            ax0.set_title('Error J')
            ax1.set_title('Accurracy')
            ax2.set_title('H evolution')
            plt.show()

        acc = self.accuracy(X)
        print "Accurracy: %f" % acc


    def backprop(self, X, Y, alpha=1, lamb=2):
        grads = self.grad(X, Y, lamb)
        self.Theta -= alpha * grads

    def forward(self, X):
        """Make the forward step, storing the value of the hypothesis
        H in each layer for backpropagation.

        I decided to store the H values (H = sigmoid(Z)) instead the Z
        as many people do for saving computation time as coping and
        assign 1 in a column (the bias term) is faster than evaluate the
        sigmoid(Z) again to compute the derivate for 2nd time.
        """
        # initial H is X + bias term
        H = np.insert(X, 0, values=1, axis=len(X.shape) > 1)  # axis 0 or 1
        self.Hs = [H]
        for Theta in self.Theta:
            # make the logistic regression for this layer
            # and store H values for backpropagation
            H = np.copy(H)
            H[:, 0] = 1

            Z = H.dot(Theta)
            Z = np.insert(Z, 0, values=1, axis=len(Z.shape) > 1)  # axis 0 or 1

            H = sigmoid(Z)     # Z --> H
            self.Hs.append(H)  # store H with all sigmoid values done

        self.H = H[:, 1:]  # remove the bias term in last step
        return self.H      # return the hypothesis

    def cost(self, X, Y, lamb=0):
        "Compute the error cost of NN"
        self.forward(X)
        H = self.H
        assert H.shape == Y.shape
        m = num_samples(Y)

        # error costs
        J = -Y * np.log(H) - (1 - Y) * np.log(1-H)
        # J = (H - Y) ** 2
        J = np.nansum(J)

        # regularization costs
        for Theta in self.Theta:
            Theta = Theta.copy()
            # TODO: review what happens if we don't clear
            Theta[0] = 0
            Theta[:, 0] = 0
            J += lamb * (Theta * Theta).sum() / 2.0

        J /= m
        self.J = J
        return J

    def grad_numerical(self, X, Y, lamb=0.0):
        "Numerical gradient by central finite difference"
        eps = 1e-4
        outputs = list()
        for t, Theta in enumerate(self.Theta):
            n, m = Theta.shape
            derivate = np.empty_like(Theta)
            t0 = time.time()
            for i in range(n):
                for j in range(m):
                    safe = Theta[i, j]
                    Theta[i, j] = safe + eps
                    f2 = self.cost(X, Y, lamb)
                    Theta[i, j] = safe - eps
                    f0 = self.cost(X, Y, lamb)
                    derivate[i, j] = (f2 - f0) / (2 * eps)

                e = time.time() - t0
                s = e / (1 + i)
                remain = (n - i) * s
                eta = t0 + remain
                eta = time.gmtime(eta)
                eta = time.strftime("%Y-%m-%d %H:%M:%S", eta)
                print "[%02d]: %03d/%03d eta: %s (%d secs)" % (t, i, n, eta, remain)

            outputs.append(derivate)

        return np.array(outputs)

    def grad(self, X, Y, lamb=0.0):
        "Compute the gradients using Back Propagation algorithm"
        # asume that a fwd has been already done
        # self.forward(X)

        outputs = list()
        delta = self.H - Y
        for i in reversed(range(len(self.Theta))):
            H = self.Hs[i]
            # we need to add the full bias term
            # so make a copy and overwrite the 1st column
            H = np.copy(H)
            H[:, 0] = 1
            DTheta = np.einsum('...i,...j->...ij', H, delta)
            if len(DTheta.shape) > 2:
                # average of all gradients per sample
                DTheta = np.average(DTheta, axis=0)

            outputs.append(DTheta)

            # we can avoid the rest of the computation
            # in the last step
            if i == 0:
                break

            # sigmoid gradient
            H = self.Hs[i]
            sigrad = H * (1 - H)

            Theta = self.Theta[i]
            delta = delta.dot(Theta.T) * sigrad
            delta = delta[:, 1:]  # remove the bias

        outputs.reverse()
        outputs = np.array(outputs)

        # regularization
        if lamb > 0.0:
            m = Y.shape[0]
            Theta = np.copy(self.Theta)
            for th in Theta:
                th[0] = 0

            outputs += Theta * lamb / m

        return np.array(outputs)

    def predict(self, X):
        "Predict the class for several samples in X"
        # TODO: use a vectorization form

        # H = self.forward(X)  # assume that has been done
        H = self.H
        predict = []
        for sample in H:
            i = np.argmax(sample)
            # p = sample[i]
            predict.append(i)
        predict = np.array(predict) + 1
        return predict

    def accuracy(self, X, Y):
        # TODO: use a vectorization form
        H = self.H
        predict = np.argmax(H, axis=1)

        match = 0
        for i, guess in enumerate(predict):
            match += Y[i, guess]

        predict = float(match) / Y.shape[0]
        return predict

    def pack(self, thetas):
        return pack(thetas, self.Theta)

    def unpack(self):
        return unpack(self.Theta)




