import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer


def load_CANCER(nb_bins):

    data = load_breast_cancer()

    feature_names, xs, ys = data.feature_names, data.data, data.target
    df = pd.DataFrame(data=xs, columns=feature_names)
    nb_points = df.shape[0]

    df_binarized = pd.DataFrame()
    questions = []
    for column in df.columns:

        bins = pd.qcut(df[column], q=nb_bins)
        cut_values = [bin.left for bin in bins.cat.categories]
        for cut_value in cut_values[1:]:
            new_col = np.zeros(nb_points, dtype=bool)
            new_col[df[column] < cut_value] = 0
            new_col[df[column] >= cut_value] = 1

            short_name = f'{column}_{cut_value:.2f}'
            questions.append(f'{column} larger than {cut_value:.2f}')

            df_binarized[short_name] = new_col

    xs = df_binarized.values
    return xs, ys, questions