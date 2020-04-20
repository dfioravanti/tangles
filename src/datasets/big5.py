
import numpy as np
import pandas as pd


def load_BIG5(path):

    df = pd.read_csv(path, header=0, sep='\t')
    df = df.drop(['race', 'age', 'engnat', 'gender', 'hand', 'source', 'country'], axis=1)

    xs = df.to_numpy()
    idx = np.random.choice(len(xs), size=300, replace=False)
    xs = xs[idx]
    ys = None

    return xs, ys
