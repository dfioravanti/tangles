import numpy as np
from sklearn.metrics.pairwise import manhattan_distances

import src.acceptance_functions
from src.algorithms import basic_algorithm
from src.config import PREPROCESSING_NO, PREPROCESSING_MAKE_SUBMODULAR
from src.config import ALGORITHM_BASIC
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


def compute_tangles(previous_tangles, current_cuts, acceptance_function, algorithm):
    if algorithm.name == ALGORITHM_BASIC:
        tangles = basic_algorithm(previous_tangles=previous_tangles,
                                  current_cuts=current_cuts,
                                  acceptance_function=acceptance_function)

    return tangles


def mask_points_in_tangle(tangle, all_cuts, threshold):

    points = all_cuts[tangle.get_idx_cuts()].T
    center = np.array(tangle.get_orientations())
    center = center.reshape(1, -1)

    distances = manhattan_distances(points, center)
    mask = distances <= threshold
    return mask


def compute_clusters(tangles, cuts, tolerance=0.8):

    predictions = []

    for tangle in tangles:
        threshold = np.int(np.trunc(len(tangle) * (1-tolerance)))
        mask = mask_points_in_tangle(tangle, cuts, threshold)
        predictions.append(mask)

    return predictions
