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


def linear_model_bias(x, w, b):
    w_array = np.array(w)
    x_array = np.array(x)
    return np.dot(w_array, x_array) + b


def main():
    n = 4
    w = [1] * n
    x = 2 * nr.random(n) - 1
    b = 0
    result = linear_model_bias(x, w, b)
    print(result)


if __name__ == '__main__':
    main()
