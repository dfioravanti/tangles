import numpy as np
import pandas as pd
import yaml


def load_MUSHROOMS(path_csv, path_yaml):

    data = np.array(pd.read_csv(path_csv, header=0))
    xs = np.zeros_like(data)

    with open(path_yaml, 'r') as f:
        names = yaml.load(f, Loader=yaml.Loader)

    for row_index, row in enumerate(data):
        for col_index, col in enumerate(row):
            if col == "?":
                col = "m"
            xs[row_index, col_index] = names["mapping"][col_index][col][0]


    ys = xs[:, 0]
    xs = xs[:, 1:]

    rows, cols = xs.shape

    idx = np.random.choice(rows, 2000)

    xs = xs[idx]
    ys = ys[idx]

    binary_xs = binarize(xs.T).T

    return binary_xs, ys


def binarize(xs):
    binary_xs = []

    for i_row, row in enumerate(xs):
        bounds = max(row)
        for answer in range(1, bounds):
            binary_xs.append(row == answer)

    return np.array(binary_xs)
