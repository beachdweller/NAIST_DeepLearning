"""
Deep Learning and Neural Networks

Advanced Research Seminar I/III
Graduate School of Information Science
Nara Institute of Science and Technology
January 2014
Instructor:
Kevin Duh, IS Building Room A-705
Office hours: after class, or appointment by email (x@is.naist.jp where x=kevinduh)

http://cl.naist.jp/~kevinduh/a/deep2014/
"""

import numpy as np
import numpy.random as nr
from sklearn.datasets.samples_generator import make_blobs
import pylab

def linear_model(w, x):
    """
    y = wT x
    :param x: data to fit. [1 x (len(x))]
    :param w: weight to fit the data. [1 x (len(x) + 1)]
    :return:
    """
    w_array = np.array(w)
    x_array = np.concatenate((x, [1.0]))
    return np.dot(w_array, x_array)


def sigmoid_z(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_x(w, x):
    """
    :param x: data to fit. [1 x (len(x))]
    :param w: weight to fit the data. [1 x (len(x) + 1)]
    :return:
    """
    return sigmoid_z(linear_model(w, x))


def d_sigmoid_dz(z):
    return sigmoid_z(z) * (1-sigmoid_z(z))


def d_sigmoid_dx(x, w):
    return d_sigmoid_dz(linear_model(x, w))


def generate_training_data(n_samples=50):
    """
    random points
    :param n_samples:
    :return:
    """
    X, Y = make_blobs(n_samples=n_samples, centers=2, random_state=0, cluster_std=0.60)
    return X, Y


def contour_sigmoid_2d(w, X, Y, filename=False, title=False):
    """
    :param w: weight as training result [1 x 3]
    :param X: training data, [n x 2]
    :param Y: training label, [n x 1]
    :return:
    """
    # plot the line, the points, and the nearest vectors to the plane
    
    contour_resolution = 20

    xx = np.linspace(-1, 5, contour_resolution)
    yy = np.linspace(-1, 5, contour_resolution)

    X1, X2 = np.meshgrid(xx, yy)
    Z = np.empty(X1.shape)
    for (i, j), x1 in np.ndenumerate(X1):
        x2 = X2[i, j]
        Z[i, j] = sigmoid_x(w, [x1, x2])
    levels = (0.1, 0.5, 0.9)
    pylab.contour(X1, X2, Z, levels)
    pylab.scatter(X[:, 0], X[:, 1], c=Y, cmap=pylab.cm.Paired)
    pylab.axis('tight')

    if not title:
        pylab.title('[%g, %g, %g] %g' % (w[0], w[1], w[2], loss_function(w, X, Y)))
    else:
        pylab.title(title)

    if filename:
        pylab.savefig(filename,dpi=300)
    else:
        pylab.show()


def loss_function(w, X, Y):
    result = 0.0
    for x, y in zip(X, Y):
        result += (sigmoid_x(w, x) - y) ** 2
    result *= 0.5

    return result


def get_sample_format(x_array):
    """
    generate format string for n samples

    example
    =======
    >>> X = np.zeros((100, 100))
    >>> get_sample_format(X)
    %03d

    :param x_array: [n x m]
    :return:
    """
    return '%0'+str(int(np.log10(x_array.shape[0])+1))+'d'


def stochastic_gradient_descent(x_array, y_array, gamma, w0=False, heuristic=True, filename_prefix=False, b_verbose=False):
    """
    try to find w minimizing the loss function through sample by sample iteration
    :param w0: initial weight. list. [(len(x) + 1) x 1]
    :param x_array: [n_sample x len(x)]
    :param y_array: [n_sample x 1]
    :param gamma: learning rate, scala, 0 < gamma
    :param heuristic: heuristic learning rate, bool, If True, gamma(k) = gamma/k
    :return:
    """
    if not w0:
        w0 = np.ones(x_array.shape[1]+1)

    if filename_prefix:
        filename_format_string = filename_prefix+get_sample_format(x_array)+'.png'

    w = w0
    counter = 1
    w_list = [w]
    for x, y in zip(x_array, y_array,):
        error = sigmoid_x(w, x) - y
        factor = ((-gamma)*error*d_sigmoid_dx(w, x))
        w[:-1] += factor * x
        w[-1] *= factor
        w_list.append(w)
        if b_verbose:
            print ("loss function = %g" % loss_function(w, x_array, y_array))

        if filename_prefix:
            contour_sigmoid_2d(w, x_array, y_array, filename_format_string % counter)
            pylab.clf()

        counter += 1
        if heuristic:
            gamma *= (1.0/counter)
    return w_list


def main():
    n = 4
    w = [1] * (n + 1)
    x = 2 * nr.random(n) - 1
    result = linear_model(w, x)
    print(result)

    w_2d = [1, 1, 1]
    X, Y = generate_training_data()
    print ("loss function = %g" % loss_function(w_2d, X, Y))

    w_list = stochastic_gradient_descent(X, Y, 1, heuristic=False, b_verbose=True, filename_prefix='blob')
    contour_sigmoid_2d(w_list[-1], X, Y)


if __name__ == '__main__':
    main()
