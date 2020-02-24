import numpy as np

from src.algorithms import exponential_algorithm
from src.config import PREPROCESSING_NO
from src.config import ALGORITHM_EXPONENTIAL


def compute_cuts(xs, preprocessing):
    if preprocessing.name == PREPROCESSING_NO:
        cuts = (xs == True).T

    return cuts


def compute_tangles(xs, cuts, algorithm):
    if algorithm.name == ALGORITHM_EXPONENTIAL:
        tangles = exponential_algorithm(xs, cuts)

    return tangles


def process(tangle):

    d = len(tangle)
    idx = np.zeros(d, dtype=int)
    orr = np.zeros(d, dtype=bool)

    for i, s in enumerate(tangle):
        if s > 0:
            idx[i] = s - 1
            orr[i] = True
        else:
            idx[i] = -s - 1
            orr[i] = False

    return idx, orr


def compute_clusters(xs, tangles, tollerance=0.8):

    predicitions = []

    for t in tangles:
        threshold = np.int(np.trunc(len(t) * tollerance))

        idx, orr = process(t)
        n_similarities = np.sum(xs[:, idx] == orr, axis=1)
        predicitions.append(n_similarities >= threshold)

    return predicitions
