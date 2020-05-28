import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer


def load_CANCER(nb_bins):

    data = load_breast_cancer()
    feature_names, xs, ys = data.feature_names, data.data, data.target
    
    return xs, ys