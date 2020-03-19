import numpy as np
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics import homogeneity_completeness_v_measure

from src.config import PREPROCESSING_NO, PREPROCESSING_MAKE_SUBMODULAR, \
    PREPROCESSING_NEIGHBOURHOOD_CUTS
from src.config import ALGORITHM_CORE
from src.algorithms import core_algorithm
from src.preprocessing import make_submodular, neighbourhood_cover


def compute_cuts(xs, preprocessing):
    if preprocessing.name == PREPROCESSING_NO:
        cuts = (xs == True).T
    elif preprocessing.name == PREPROCESSING_MAKE_SUBMODULAR:
        cuts = (xs == True).T
        cuts = make_submodular(cuts)
    elif preprocessing.name == PREPROCESSING_NEIGHBOURHOOD_CUTS:
        cuts = neighbourhood_cover(A=xs, nb_common_neighbours=9, max_k=50)
    return cuts


def order_cuts(cuts, order_function):

    cost_cuts = np.zeros(len(cuts))

    for i_cut, cut in enumerate(cuts):
        cost_cuts[i_cut] = order_function(cut)

    idx = np.argsort(cost_cuts)

    return cuts[idx], cost_cuts[idx]


def compute_tangles(tangles, current_cuts, idx_current_cuts, min_size, algorithm):
    if algorithm.name == ALGORITHM_CORE:
            tangles = core_algorithm(tangles, current_cuts, idx_current_cuts, min_size)

    return tangles


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
