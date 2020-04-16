import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph


def load_BIG5(path):

    df = pd.read_csv(path, header=0, sep='\t')
    df = df.drop(['race', 'age', 'engnat', 'gender', 'hand', 'source', 'country'], axis=1)

    name_columns = df.columns.values
    first_char = lambda s: s[0]
    f = np.vectorize(first_char)
    ys_cat = f(name_columns)
    ys = np.zeros_like(ys_cat, dtype=int)

    ys[ys_cat == 'O'] = 0
    ys[ys_cat == 'C'] = 1
    ys[ys_cat == 'E'] = 2
    ys[ys_cat == 'A'] = 3
    ys[ys_cat == 'N'] = 4

    ys = np.repeat(ys, 5)

    xs = df.to_numpy()
    xs = xs.T

    # binary_xs = binarize(xs)[:, np.random.choice(np.arange(xs.shape[1]), 2000, False)]
    binary_xs = binarize(xs)

    A = kneighbors_graph(binary_xs.T, 5, metric='hamming').toarray().astype(int)
    G = nx.from_numpy_matrix(A)

    return binary_xs.T, ys, A, G


def binarize(xs):
    rows, cols = xs.shape
    binary_xs = np.zeros([rows * 5, cols])

    for i_row, row in enumerate(xs):
        for answer in range(5):
            binary_xs[i_row*5 + answer] = row >= answer

    return binary_xs