import numpy as np
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics import homogeneity_completeness_v_measure

from src.config import PREPROCESSING_FEATURES, PREPROCESSING_MAKE_SUBMODULAR, \
    PREPROCESSING_NEIGHBOURHOOD_CUTS, PREPROCESSING_KARGER, PREPROCESSING_FAST_MINCUT
from src.config import ALGORITHM_CORE
from src.tangles import core_algorithm
from src.cuts import make_submodular, cuts_from_neighbourhood_cover, find_approximate_mincuts


def compute_cuts(xs, preprocessing):

    """
    Given a set of points or an adjacency matrix this function returns the set of cuts that we will use
    to compute tangles.
    This different types of preprocessing are available

     1. PREPROCESSING_FEATURES: consider the features as cuts
     2. PREPROCESSING_MAKE_SUBMODULAR: consider the features as cuts and then make them submodular
     3. PREPROCESSING_NEIGHBOURHOOD_CUTS: Given an adjacency matrix build a cover of the graph and use that as starting
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
    elif preprocessing.name == PREPROCESSING_MAKE_SUBMODULAR:
        cuts = (xs == True).T
        cuts = make_submodular(cuts)
    elif preprocessing.name == PREPROCESSING_NEIGHBOURHOOD_CUTS:
        cuts = cuts_from_neighbourhood_cover(A=xs, nb_common_neighbours=1, max_k=6)
    elif preprocessing.name == PREPROCESSING_KARGER:
        cuts = find_approximate_mincuts(A=xs, nb_cuts=preprocessing.nb_cuts, algorthm='karger')
    elif preprocessing.name == PREPROCESSING_FAST_MINCUT:
        cuts = find_approximate_mincuts(A=xs, nb_cuts=preprocessing.nb_cuts, algorthm='fast')

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


# Old code. But it might be useful later

def mask_points_in_tangle(tangle, all_cuts, threshold):

    points = all_cuts[tangle.get_idx_cuts()].T
    center = np.array(tangle.get_orientations())
    center = center.reshape(1, -1)

    distances = manhattan_distances(points, center)
    mask = distances <= threshold
    return mask.reshape(-1)


def compute_clusters(tangles, cuts, tolerance=0.8):

    n_cuts, n_points = cuts.shape

    predictions = np.zeros(n_points, dtype=int)

    for i, tangle in enumerate(tangles, 1):
        len_tangle = len(tangle)

        numpy_tangle = np.zeros((len_tangle, n_cuts), dtype=bool)
        for j, oriented_cut in enumerate(tangle):
            numpy_tangle[j, :] = oriented_cut.to_numpy(n_cuts)

        threshold = np.int(np.trunc(len_tangle * (1-tolerance)))
        mask = mask_points_in_tangle(tangle, cuts, threshold)
        predictions[mask] = i

    return predictions


def compute_evaluation(ys, predictions):

    evaluations = {}

    for order, prediction in predictions.items():
        evaluations[order] = {}

        # Save the number of points that do not belong to a tangle
        evaluations[order]["unassigned"] = np.sum(prediction == 0)

        mask_assigned = prediction != 0
        homogeneity, completeness, v_measure_score = \
            homogeneity_completeness_v_measure(ys[mask_assigned], prediction[mask_assigned])
        evaluations[order]["homogeneity"] = homogeneity
        evaluations[order]["completeness"] = completeness
        evaluations[order]["v_measure_score"] = v_measure_score

    return evaluations
