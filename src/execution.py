import numpy as np
from sklearn.metrics.pairwise import manhattan_distances

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


def mask_points_in_tangle(xs, tangle, threshold):

    distances = manhattan_distances(xs[:, tangle.cuts], tangle.orientations.reshape(1, -1))
    mask = distances <= threshold
    return mask


def compute_clusters(xs, tangles, tolerance=0.8):

    predictions = []

    for tangle in tangles:
        p = []
        for t in tangle:
            threshold = np.int(np.trunc(t.size * (1-tolerance)))
            mask = mask_points_in_tangle(xs, t, threshold)
            p.append(mask)
        predictions.append(p)

    return predictions
