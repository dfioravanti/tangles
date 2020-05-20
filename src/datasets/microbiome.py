import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph


def load_MICROBIOME(path, k):

    df = pd.read_csv(path, header=0, sep=";")
    name_columns = df.columns.values

    ys = np.argmax(df[name_columns[:7]].to_numpy(), axis=1)
    xs = df[name_columns[8:]].to_numpy()

    A = kneighbors_graph(xs, k, mode="distance").toarray()

    sigma = np.median(A[A > 0])

    A[A > 0] = np.exp(- A[A > 0]/(2 * sigma**2))

    return xs, ys, A
