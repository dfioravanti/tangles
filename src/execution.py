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
        cuts = (xs == True).T
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


def mask_points_in_tangle(xs, tangle, cuts, threshold):

    distances = manhattan_distances(cuts[tangle.cuts].T, tangle.orientations.reshape(1, -1))
    mask = distances <= threshold
    return mask


def compute_clusters(xs, tangles, cuts, tolerance=0.8):

    predictions = []

    for tangle in tangles:
        threshold = np.int(np.trunc(tangle.size * (1-tolerance)))
        mask = mask_points_in_tangle(xs, tangle, cuts, threshold)
        predictions.append(mask)

    return predictions


def fix_indexes(tangles, indexes):
    indexes = np.array(indexes)
    for tangle in tangles:
        tangle.cuts = indexes[tangle.cuts]
