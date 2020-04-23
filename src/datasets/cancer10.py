import numpy as np
import pandas as pd


def load_CANCER10(path):

    df = pd.read_csv(path, header=0)
    name_columns = df.columns.values

    ys = np.array(df["class"])
    xs = df[name_columns[:-1]]

    xs = xs.to_numpy()

    return xs, ys


