import numpy as np
import matplotlib.pyplot as plt
import plot_activation as pa


def main(min_x=-8, max_x=8):
    x_array = np.linspace(min_x, max_x, 101)

    process_them = [
        {'label': 'sigmoid', 'f': pa.get_sigmoid_function_sympy()},
    ]

    for fmt in ('pdf', 'png'):
        for d in process_them:
            pa.proc_sympy_function(x_array, d['f'], d['label'])

        plt.grid(True)
        plt.legend(loc=0)
        plt.axis('equal')
        plt.savefig('sigmoid.%s' % fmt)


if __name__ == '__main__':
    main()
