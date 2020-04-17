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

    df_binarize = pd.DataFrame()
    for column in df.columns.values:
        for i in range(2, 6):
            new_col = df[column].copy()
            new_col[:] = 0
            new_col[df[column].between(i, 5)] = 1
            name = f'{new_col.name}_{i}_5'

            df_binarize[name] = new_col

    xs = df_binarize.to_numpy().astype(bool)
    idx = np.random.choice(len(xs), size=300, replace=False)
    xs = xs[idx]
    ys = None

    return xs, ys
