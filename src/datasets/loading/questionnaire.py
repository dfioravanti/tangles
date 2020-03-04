import pathlib

import numpy as np


def load_synthetic_questionnaire(path):

    path = pathlib.Path(path)

    xs = np.loadtxt(path / "xs.txt", dtype="bool")
    ys = np.loadtxt(path / "ys.txt", dtype="int")
    cs = np.loadtxt(path / "cs.txt", dtype="bool")

    return xs, ys, cs
