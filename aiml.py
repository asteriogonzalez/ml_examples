"""Some convenient functions for all exercises
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
from scipy.optimize import minimize

from timeit import timeit

# TODO: check the extra column in Thea when Theta is optimized and from scartch
# TODO: BP with regularization

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
    error = [cost(t, *args) for t in trajectory]
    plt.plot(error)
    plt.ylabel('Error')
    plt.xlabel('Iterations')
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

def  solve_logistic_regression(X, y, theta=None, learning_rate=1.0):
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


def strip_bias(H):
    if len(H.shape) > 1:
        return H[:, 1:]
    return H[1:]

def num_samples(H):
    if len(H.shape) > 1:
        return H.shape[0]
    return 1


class FNN(object):
    def __init__(self, sizes=None):
        self.Theta = list()
        self.Z = list()  # intermediate Z output values for backprop
        self.H = None    # the last hypothesis value

        if sizes:
            self.create_netwotk(sizes)

    def create_netwotk(self, *sizes):
        """Create the network structure.
        sizes contains the number of nodes in each layer.

        size[0] is the number of input features, without bias term
        size[1] is the number of nodes (sub-features) witous bias
        ...
        size[n] is the number of nodes at the end, usually match with
                the number of classes in classification problems.
        """
        self.Theta = list()
        for i in range(0, len(sizes) - 1):
            # use extended sizes with bias term for convenience
            # but remove in the last stage
            size = (sizes[i] + 1, sizes[i+1] + 1)
            # create and initialize layer with random values
            # following the advice from Andre Ng
            epsilon = np.sqrt(6) / np.sqrt(sum(size))
            Theta = np.random.randn(*size) * epsilon
            self.Theta.append(Theta)

        # remove the bias term in the last stage, as we
        # haven't anything to compare with
        self.Theta[-1] = Theta[:, 1:]

    def setup(self, X, Y, Theta=None):
        if Theta is not None:
            # add a column in Theta_i that yield a 1.0 in H_(i+1)
            # avoiding to insert a 'one' column in all steps
            # (searching a compact vectorized way to implement FF and BP)
            # I don't add this colum in the last layer due some 'nan' will
            # appears in (1-Y) * np.log(1-H)
            # even 1-Y is zero, log(1-H) = log(0) = inf in this terms
            # This way saves to add an extra column in Y as well
            # M = 1e6
            # for i in range(len(Theta) - 1):
                # Theta[i] = np.insert(Theta[i], 0, values=M, axis=1)  # axis 0 or 1

            # use np arrays for update all Thetas at the same time
            self.Theta = np.array(Theta)

        # H = np.insert(X, 0, values=1, axis=len(X.shape) > 1)  # axis 0 or 1
        # self._forward(H)  # to compute the right shapes

        # add the backprop bias term for speed up
        # (This way saves to add an extra column in Y as well, see above)
        # Y = np.insert(Y, 0, values=1, axis=len(Y.shape) > 1)  # axis 0 or 1
        self.Y = Y

    def reset_H(self):
        self.H = np.empty((len(self.Theta) + 1))

    def solve(self, X, y, alpha=0.5, lam=0.0):
        self.X = X
        # setup optimization params
        klasses = np.unique(y)
        n_klasses = klasses.size
        klasses = range(n_klasses)
        samples, features = X.shape
        y.shape = (samples, )

        # setup the Y multiclass matrix
        Y = np.zeros((samples, n_klasses), dtype=np.int8)
        for i, klass in enumerate(y):
            Y[i][klass - 1] = 1

        self.setup(X, Y)

        errors = []
        deltas = []

        sample = 3000
        yy = Y[sample]

        H = self.H[0]
        last_acc = 0.0
        for iteration in xrange(9999):
            self.forward(X)
            J = self._cost(lam)
            self._backprop(alpha, lam)

            delta = np.mean([d.mean() for d in self.H])
            if delta < 1e-3 or J < 1e-5:
                print "-End-"
                break

            if not(iteration % 10):
                p = self.predict(X)
                acc =  (p == y).mean()

                print "[%4d] Accurracy: %f, Cost: J=%s, delta=%s, alpha=%f" % \
                              (iteration, acc, J, delta, alpha)

                deltas.append(delta)
                errors.append(J)
                if acc <= last_acc:
                    print "Max Accurracy: %f : STOP" % last_acc
                    break


        f, (ax0, ax1)= plt.subplots(2, sharex=True)
        ax0.plot(errors)
        ax1.plot(deltas)
        ax0.set_title('Error J')
        ax1.set_title('H evolution')
        plt.show()

        predict = self.Hs[-1][sample]
        print predict
        print y[sample]


        p = self.predict(X)
        print "Accurracy: %f" % (p == y).mean()

    def forward(self, X):
        # H = np.insert(X, 0, values=1, axis=len(X.shape) > 1)  # axis 0 or 1
        # Note  that each H and Theta are a marrices with the bias term included.
        self._forward(X)
        return self.H

    def _forward(self, H):
        H = np.insert(H, 0, values=1, axis=len(H.shape) > 1)  # axis 0 or 1
        self.Zs = [H]
        self.Hs = [H]
        for i, Theta in enumerate(self.Theta):
            # make the logistic regression for this layer.
            H = np.copy(H)
            H[:, 0] = 1

            Z = H.dot(Theta)
            Z = np.insert(Z, 0, values=1, axis=len(Z.shape) > 1)  # axis 0 or 1

            H = sigmoid(Z)   # Z --> H
            self.Hs.append(H)
            self.Zs.append(Z)

        self.H = H[:, 1:]

    def _backprop(self, alpha=1, lam=0.01):
        grads = self._BP_gradients(lam)
        # grads = self._gradients(self.X, self.Y)
        self.Theta -= alpha * grads
        foo = 1

    def _gradients(self, X, Y):
        eps = 1e-4
        outputs = list()
        for Theta in self.Theta:
            n, m = Theta.shape
            derivate = np.empty_like(Theta)
            for i in range(n):
                print i
                for j in range(m):
                    safe = Theta[i, j]
                    Theta[i, j] = safe + eps
                    f2 = self.cost(X, Y)
                    Theta[i, j] = safe - eps
                    f0 = self.cost(X, Y)
                    derivate[i,j] = (f2 - f0) / (2 * eps)

            outputs.append(derivate)

        return np.array(outputs)

    def _BP_gradients(self, lam=0.0):
        outputs = list()
        dd = self.H - self.Y
        for i in reversed(range(len(self.Theta))):
            H = self.Hs[i]
            # we need to add the full bias term from dd
            # so make a copy and overwrite the 1st column
            H = np.copy(H)
            H[:, 0] = 1
            dTheta = np.einsum('...i,...j->...ij', H, dd)
            if len(dTheta.shape) > 2:
                dTheta = np.average(dTheta, axis=0)
            outputs.append(dTheta)

            # we can avoid the following computation
            # in the last step
            if i == 0:
                break

            H = self.Hs[i]
            # sigmoid gradient
            sigrad = H * (1 - H)

            Theta = self.Theta[i]
            dd = dd.dot(Theta.T) * sigrad
            dd = dd[:, 1:]


            # NOTE: dd[0] is always zero, so I implement a version with
            # NOTE: extended Theta columns to avoid insert new columns
            # NOTE: and row inside loops, in order to find a full
            # NOTE: verctorized backprop algorithm without loops
            # NOTE: see BP in a full connected layer without loops
            # https://iamtrask.github.io/2015/07/12/basic-python-network/

            # TODO: implement regularization, here or in a new function

        outputs.reverse()
        outputs = np.array(outputs)

        # regularization
        if lam > 0.0:
            m = self.Y.shape[0]
            Theta = np.copy(self.Theta)
            for th in Theta:
                th[0] = 0

            outputs += Theta * lam / m



        return np.array(outputs)


    def derivate(self, i):
        "In the sigmoid case, is a straight value"
        Z = self.H[i]
        return Z * (1 - Z)

    def predict(self, X):
        H = self.forward(X)
        predict = []
        for sample in H:
            i = np.argmax(sample)
            # p = sample[i]
            predict.append(i)
        predict = np.array(predict) + 1
        return predict

    def cost(self, X, Y, lam=0):
        self.forward(X)
        return self._cost(lam)

    def _cost(self, lam=0):
        H = self.H
        Y = self.Y
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
            J += lam * (Theta * Theta).sum() / 2.0

        J /= m
        return J

    def _d_show_info(self):
        for i, H in enumerate(self.H):
            if i < len(self.Theta):
                Theta = self.Theta[i]
                print 'H%i: %s\t-->\tT%i: %s' % (i, H.shape, i, Theta.shape)
            else:
                print 'H%i: %s' % (i, H.shape)


        foo = 1




