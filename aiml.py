import numpy as np

def H(theta, X):
    "Linear regression Hypotesis"
    return np.dot(X, theta)

def J(theta, X, y):
    "The cost funtcion"
    d = H(theta, X) - y
    return np.dot(d, d) / (2 * len(y))

def gradient_descent(theta, X, y, alpha=0.01, iters=20000, err=0.00001, evolution=False):
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
            print('[%03d] err: %0.5f' % (i, J))

        if (last_err - J) < err:
            break
        last_err = J
        if evolution:
            progress.append(last_err)

    return theta, progress

