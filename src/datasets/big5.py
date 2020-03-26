import numpy as np
import pandas as pd


def load_BIG5(path):

    df = pd.read_csv(path, header=0, sep='\t')
    df = df.drop(['race', 'age', 'engnat', 'gender', 'hand', 'source', 'country'], axis=1)
    df = df.astype('category')
    binarized_xs = pd.get_dummies(df).astype('bool').to_numpy()

    nb_users, nb_features = binarized_xs.shape
    compressed_xs = np.zeros((nb_users, nb_features // 5 * 3), dtype=bool)

    i = i_compressed = 0
    while i < nb_features:
        if i % 5 == 0 or i % 5 == 3:
            compressed_xs[:, i_compressed] = binarized_xs[:, i] + binarized_xs[:, i+1]
            i += 2
        else:
            compressed_xs[:, i_compressed] = binarized_xs[:, i]
            i += 1
        i_compressed += 1

    xs = df.to_numpy()
    idx = np.random.choice(len(xs), size=2000, replace=False)
    xs = xs[idx].T
    ys = np.zeros(len(xs), dtype=int)

    return xs, ys
