import sympy as sp


def plot_sigmoid(x_min=-8, x_max=8):
    sigmoid, x = get_sigmoid_function()

    integrated_sigmoid = sp.integrate(sigmoid, x)
    print('integrated sigmoid = %s' % integrated_sigmoid)

    sp.plot(sigmoid, integrated_sigmoid, (x, x_min, x_max))


def get_sigmoid_function():
    x = sp.symbols('x')
    sigmoid = 1 / (1 + sp.exp(-x))
    return sigmoid, x


if __name__ == '__main__':
    plot_sigmoid()
