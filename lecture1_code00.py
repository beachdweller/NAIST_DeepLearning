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


def main():
    n = 4
    w = [1] * (n + 1)
    x = 2 * nr.random(n) - 1
    result = linear_model(w, x)
    print(result)


if __name__ == '__main__':
    main()
