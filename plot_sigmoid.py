import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def get_integrated_sigmoid_sympy():
    sigmoid, x = get_sigmoid_function_sympy()

    return [x, sp.integrate(sigmoid, x), 'numpy']


def get_sigmoid_function_sympy():
    x = sp.symbols('x')
    sigmoid = 1 / (1 + sp.exp(-x))
    return [x, sigmoid, 'numpy']


def main(min_x=-8, max_x=8):
    # Numeric Computation, Sympy documentation, 2016, [Online] Available
    # http://docs.sympy.org/dev/modules/numeric-computation.html
    sigmoid = sp.lambdify(*get_sigmoid_function_sympy())

    x = np.linspace(min_x, max_x)
    y = sigmoid(x)
    plt.plot(x, y)

    plt.grid(True)
    plt.savefig('sigmoid.png', dpi=300)


if __name__ == '__main__':
    main()
