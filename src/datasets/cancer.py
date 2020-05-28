import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer


def load_CANCER(nb_bins):

    data = load_breast_cancer()
    feature_names, xs, ys = data.feature_names, data.data, data.target
    
    if nb_bins > 0:
        df = pd.DataFrame(data=xs, columns=feature_names)
        df_bined = pd.DataFrame()
        for column in df.columns:
            df_bined[column] = pd.qcut(df[column], q=nb_bins, labels=False) + 1

        xs = df_bined.values
    return xs, ys