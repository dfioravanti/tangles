import pathlib
from functools import partial

import numpy as np

from src.config import DATASET_QUESTIONNAIRE_SYNTHETIC
from src.order_functions import order_questionnaire


def get_dataset(dataset):

    if dataset.type == DATASET_QUESTIONNAIRE_SYNTHETIC:
        xs, ys, cs = load_synthetic_datataset(dataset.path)
        order_function = partial(order_questionnaire, xs, 20)

    return xs, ys, order_function


def load_synthetic_datataset(path):

    path = pathlib.Path(path)

    xs = np.loadtxt(path / "xs.txt", dtype="bool")
    ys = np.loadtxt(path / "ys.txt", dtype="int")
    cs = np.loadtxt(path / "cs.txt", dtype="bool")

    return xs, ys, cs