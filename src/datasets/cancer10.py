import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer


def load_CANCER10(path):

    df = pd.read_csv(path, header=0)

    ys = np.array(df["class"])

    name_columns = df.columns.values

    xs = df[name_columns[:-1]].to_numpy()

    binary_xs, repeater = binarize(xs)

    ys = np.repeat(ys, repeater)

    return binary_xs, ys


def binarize(xs):
    rows, cols = xs.shape
    bounds = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    expand = len(bounds)
    binary_xs = np.zeros([rows * expand, cols])

    for i_row, row in enumerate(xs):
        for i_answer, answer in enumerate(bounds):
            binary_xs[i_row * expand + i_answer] = row <= answer

    return binary_xs, expand