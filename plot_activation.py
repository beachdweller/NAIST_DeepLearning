import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy.abc import x


def get_integrated_tanh_sympy():
    return sp.integrate(sp.tanh(x), x)


def get_integrated_sigmoid_sympy():
    sigmoid = get_sigmoid_function_sympy()
    return sp.integrate(sigmoid, x)


def get_sigmoid_function_sympy():
    sigmoid = 1 / (1 + sp.exp(-x))
    return sigmoid


def proc_sympy_function(x_array, expression, label, lib='numpy'):
    # Numeric Computation, Sympy documentation, 2016, [Online] Available
    # http://docs.sympy.org/dev/modules/numeric-computation.html
    f = sp.lambdify(x, expression, lib)
    y_f = f(x_array)
    plt.plot(x_array, y_f, label=label)


def step(z):
    """
    return an array of same size as z.
    For elements of z larger than zero, return one.
    Otherwise zero.

    :type z: Any
    :param z: input. preferrably a ndarray
    :return:
    """
    result = np.zeros_like(z)
    result[z > 0] = 1.0
    return result


def ReLU(x_array):
    result = np.zeros_like(x_array)
    result[0 < x_array] = x_array[0 < x_array]
    return result


def main(min_x=-8, max_x=8):
    min_x, max_x = sorted([min_x, max_x])
    x_array = np.linspace(min_x, max_x, 101)

    x_array_step = get_x_step_array(min_x, max_x)

    symbols_to_be_processed_list = [
        {'label': 'sigmoid', 'f': get_sigmoid_function_sympy()},
        {'label': 'integrated sigmoid', 'f': get_integrated_sigmoid_sympy()},
    ]

    for fmt in ('pdf', 'png'):
        plt.clf()

        for d in symbols_to_be_processed_list:
            proc_sympy_function(x_array, d['f'], d['label'])

        y_ReLU = ReLU(x_array)
        plt.plot(x_array, y_ReLU, label='ReLU')

        y_step = step(x_array_step)
        plt.plot(x_array_step, y_step, label='step')

        plt.grid(True)
        plt.legend(loc=0)
        plt.axis('equal')
        plt.savefig('activation.%s' % fmt)


def get_x_step_array(min_x, max_x):
    if 0 > min_x * max_x:
        # if min_x and max_x have different signs
        x_array_step = np.array([min_x, 0.0, 1e-10, max_x])
    elif 0 < min_x * max_x:
        # if min_x and max_x have same signs
        x_array_step = np.array([min_x, max_x])
    elif 0 == min_x:
        x_array_step = np.array([min_x, 1e-10, max_x])
    elif 0 == max_x:
        x_array_step = np.array([min_x, 0, 1e-10])
    else:
        raise ValueError('Unable to decide x array')

    return x_array_step


if __name__ == '__main__':
    main()
