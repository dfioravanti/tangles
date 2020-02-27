import numpy as np
from sklearn.metrics.pairwise import manhattan_distances

from src.algorithms import exponential_algorithm
from src.config import PREPROCESSING_NO, PREPROCESSING_MAKE_SUBMODULAR
from src.config import ALGORITHM_EXPONENTIAL
from src.preprocessing import make_submodular


def compute_cuts(xs, preprocessing):
    if preprocessing.name == PREPROCESSING_NO:
        cuts = (xs == True).T
    elif preprocessing.name == PREPROCESSING_MAKE_SUBMODULAR:
        cuts = (xs == True).T.astype('B')
        cuts = make_submodular(cuts)

    return cuts


def order_cuts(cuts, order_function):

    cost_cuts = {}

    for i_cut, cut in enumerate(cuts):
        order = np.int(np.ceil(order_function(cut)))

        previous_cuts = cost_cuts.get(order)
        if previous_cuts is None:
            cost_cuts[order] = [i_cut]
        else:
            previous_cuts.append(i_cut)

    return cost_cuts


def compute_tangles(xs, cuts, algorithm):
    if algorithm.name == ALGORITHM_EXPONENTIAL:
        tangles = exponential_algorithm(xs, cuts)

    return tangles


def mask_points_in_tangle(xs, tangle, threshold):

    distances = manhattan_distances(xs[:, tangle.cuts], tangle.orientations.reshape(1, -1))
    mask = distances <= threshold
    return mask


def compute_clusters(xs, tangles, tolerance=0.7):

    predictions = []

    for tangle in tangles:
        p = []
        for t in tangle:
            threshold = np.int(np.trunc(t.size * (1-tolerance)))
            mask = mask_points_in_tangle(xs, t, threshold)
            p.append(mask)
        predictions.append(p)

    return predictions
