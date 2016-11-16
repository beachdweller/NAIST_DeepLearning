import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main(min_x=-8, max_x=8):
    x = np.linspace(min_x, max_x)
    y = sigmoid(x)
    plt.plot(x, y)

    plt.grid(True)
    plt.savefig('sigmoid.png', dpi=300)


if __name__ == '__main__':
    main()
