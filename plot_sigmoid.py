import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy.abc import x


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


def ReLU(x_array):
    result = np.zeros_like(x_array)
    result[0 < x_array] = x_array[0 < x_array]
    return result


def main(min_x=-8, max_x=8):
    x_array = np.linspace(min_x, max_x)

    process_them = [
        {'label': 'sigmoid', 'f': get_sigmoid_function_sympy()},
        {'label': 'integrated sigmoid', 'f': get_integrated_sigmoid_sympy()},
    ]

    for d in process_them:
        proc_sympy_function(x_array, d['f'], d['label'])

    y_ReLU = ReLU(x_array)
    plt.plot(x_array, y_ReLU, label='ReLU')

    plt.grid(True)
    plt.legend(loc=0)
    plt.savefig('sigmoid.png', dpi=300)


if __name__ == '__main__':
    main()
