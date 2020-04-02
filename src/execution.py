import numpy as np
from sklearn.metrics import homogeneity_completeness_v_measure

from src.config import ALGORITHM_CORE
from src.config import PREPROCESSING_FEATURES, PREPROCESSING_KMODES, PREPROCESSING_KARNIG_LIN
from src.cuts import find_kmodes_cuts, kernighan_lin
from src.tangles import core_algorithm

MISSING = -1


def compute_cuts(xs, preprocessing):
    """
    Given a set of points or an adjacency matrix this function returns the set of cuts that we will use
    to compute tangles.
    This different types of preprocessing are available

     1. PREPROCESSING_FEATURES: consider the features as cuts
     2. PREPROCESSING_MAKE_SUBMODULAR: consider the features as cuts and then make them submodular
     3. PREPROCESSING_RANDOM_COVER: Given an adjacency matrix build a cover of the graph and use that as starting
                                          point for creating the cuts

    Parameters
    ----------
    xs: array of shape [n_points, n_features] or array of shape [n_points, n_points]
        The points in our space or an adjacency matrix
    preprocessing: SimpleNamespace
        The parameters of the preprocessing

    Returns
    -------
    cuts: array of shape [n_cuts, n_points]
        The bipartitions that we will use to compute tangles
    """

    if preprocessing.name == PREPROCESSING_FEATURES:
        cuts = (xs == True).T
    elif preprocessing.name == PREPROCESSING_KARNIG_LIN:
        cuts = kernighan_lin(xs=xs,
                             nb_cuts=preprocessing.karnig_lin.nb_cuts,
                             fractions=preprocessing.karnig_lin.fractions)
    elif preprocessing.name == PREPROCESSING_KMODES:
        cuts = find_kmodes_cuts(xs=xs, max_nb_clusters=preprocessing.kmodes.max_nb_clusters)

    return cuts


def order_cuts(cuts, order_function):
    """
    Compute the order of a series of bipartitions

    Parameters
    ----------
    cuts: array of shape [n_points, n_cuts]
        The bipartitions that we want to know the order of
    order_function: function
        The order function

    Returns
    -------
    cuts: array of shape [n_cuts, n_points]
        The cuts reodered according to their cost
    cost_cuts: array of shape [n_cuts]
        The cost of the corresponding cut
    """

    cost_cuts = np.zeros(len(cuts))

    for i_cut, cut in enumerate(cuts):
        cost_cuts[i_cut] = order_function(cut)

    idx = np.argsort(cost_cuts)

    return cuts[idx], cost_cuts[idx]


def compute_tangles(tangles, current_cuts, idx_current_cuts, min_size, algorithm):
    """
    Select with tangle algorithm to use. For now only one algorithm is supported.

    Parameters
    ----------
    tangles: list
        The list of all FULL tangles computed up until now
    current_cuts: list
        The list of cuts that we need to try to add to the tangles. They are ordered by order
    idx_current_cuts: list
        The list of indexes of the current cuts in the list of all cuts
    min_size: int
        Minimum triplet size for a tangle to be a tangle.
    algorithm: SimpleNamespace
        The parameters of the algorithm

    Returns
    -------

    """

    if algorithm.name == ALGORITHM_CORE:
        tangles = core_algorithm(tangles, current_cuts, idx_current_cuts, min_size)

    return tangles


def compute_clusters(tangles, all_cuts, tolerance=0.8):
    _, n_points = all_cuts.shape

    predictions = np.zeros(n_points, dtype=int) + MISSING

    for i, tangle in enumerate(tangles):
        cuts = list(tangle.specification.keys())
        orientations = list(tangle.specification.values())

        nb_cuts_in_tangle = len(cuts)
        matching_cuts = np.sum((all_cuts[cuts, :].T == orientations), axis=1)

        threshold = np.int(np.trunc(nb_cuts_in_tangle * tolerance))
        predictions[matching_cuts >= threshold] = i

    return predictions


def compute_evaluation(ys, predictions):
    evaluation = {}
    evaluation['v_measure_score'] = None
    evaluation['order_max'] = None
    evaluation['unassigned'] = None

    for order, prediction in predictions.items():

        unassigned = np.sum(prediction == -1)

        homogeneity, completeness, v_measure_score = \
            homogeneity_completeness_v_measure(ys, prediction)

        if evaluation['v_measure_score'] is None or evaluation['v_measure_score'] < v_measure_score:
            evaluation["homogeneity"] = homogeneity
            evaluation["completeness"] = completeness
            evaluation["v_measure_score"] = v_measure_score
            evaluation['order_max'] = order
            evaluation['unassigned'] = unassigned

    return evaluation
