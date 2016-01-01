import sympy as sp
import numpy as np


def matrix(s, n, m):
    # ref: Brendan Wood, MRocklin, Amelio Vazquez-Reina, Automatically populating matrix elements in SymPy, http://stackoverflow.com/questions/6877061/automatically-populating-matrix-elements-in-sympy 2013 Dec 17 (Accessed 2016 01 01).
    return sp.Matrix(n, m, lambda i,j:sp.var('%s_%d%d' % (s, i+1,j+1)))


def sigmoid_z(z):
    z_array = np.array(z)
    n_row, m_column = z_array.shape
    z_list = z_array.tolist()

    result = []
    for i in range(n_row):
        new_row = []
        z_row = z_list[i]
        for j in range(m_column):
            new_row.append(sp.simplify(1.0 / (1 + sp.exp(-z_row[j]))))
        result.append(new_row)

    return result


def main():
    i = 4
    j = 3
    k = 2

    xi = matrix('xi', i, 1)
    wij = matrix('wij', i, j)
    wjk = matrix('wjk', j, k)

    in_k = sp.simplify(wjk.T * wij.T * xi)
    print(in_k.shape)

    y_k = sigmoid_z(in_k)
    print(y_k)


if __name__ == '__main__':
    main()
