import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
import networkx as nx


def load_MICROBIOME(path, k, use_questions=False, dual=False):
    if use_questions:
        return load_questions_as_cut(path, k)
    if dual:
        return load_dual_data(path, k)
    else:
        return load_epsilin_graph(path, k)


def load_questions_as_cut(path, bins):
    df = pd.read_csv(path, header=0, sep=";")
    name_columns = df.columns.values

    ys = np.argmax(df[name_columns[:7]].to_numpy(), axis=1)
    xs = df[name_columns[8:]].to_numpy()

    xs = xs / np.max(xs, axis=0) * bins

    return xs, ys, None, None


def load_dual_data(path, bins):
    df = pd.read_csv(path, header=0, sep=";")
    name_columns = df.columns.values

    ys = np.argmax(df[name_columns[:7]].to_numpy(), axis=1)
    xs = df[name_columns[8:]].to_numpy().T

    xs = xs / np.max(xs, axis=0) * bins

    return xs, ys, None, None


def load_epsilin_graph(path, k):
    df = pd.read_csv(path, header=0, sep=";")
    name_columns = df.columns.values

    ys = np.argmax(df[name_columns[:7]].to_numpy(), axis=1)
    xs = df[name_columns[8:]].to_numpy()

    A = kneighbors_graph(xs, k, mode="distance").toarray()

    sigma = np.median(A[A > 0])

    A[A > 0] = np.exp(- A[A > 0]/(2 * sigma**2))

    G = nx.from_numpy_array(A)

    return xs, ys, A, G
