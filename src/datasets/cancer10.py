import numpy as np
import pandas as pd
import networkx as nx


def load_CANCER10(path):

    df = pd.read_csv(path, header=0)
    name_columns = df.columns.values

    ys = np.array(df["class"])
    xs = df[name_columns[:-1]]

    xs = xs.to_numpy()

    #binary_xs, repeater = binarize(xs.T)

    #return binary_xs.T, ys

    return xs, ys


def binarize(xs):
    rows, cols = xs.shape
    bounds = np.arange(1, 10, 1)
    expand = len(bounds)
    binary_xs = np.zeros([rows * expand, cols])

    for i_row, row in enumerate(xs):
        for i_answer, answer in enumerate(bounds):
            binary_xs[i_row * expand + i_answer] = row <= answer

    return binary_xs, expand