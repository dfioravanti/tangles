from copy import deepcopy

import numpy as np
from sklearn.metrics import homogeneity_completeness_v_measure

from src.config import ALGORITHM_CORE
from src.config import PREPROCESSING_FEATURES, PREPROCESSING_KMODES, PREPROCESSING_KARNIG_LIN
from src.cuts import find_kmodes_cuts, kernighan_lin
from src.loading import get_dataset_and_order_function
from src.plotting import plot_cuts, plot_heatmap_graph, plot_heatmap
from src.tangles import core_algorithm


def compute_cuts(xs, preprocessing, verbose):
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
                             fractions=preprocessing.karnig_lin.fractions,
                             verbose=verbose)
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


def compute_clusters(tangles_by_orders, all_cuts, verbose):

    predictions_by_order = {}

    for order, tangles in tangles_by_orders.items():

        if verbose >= 2:
            print(f"\tCompute clusters for order {order}", flush=True)
        _, n_points = all_cuts.shape
        nb_tangles = len(tangles)

        matching_cuts = np.zeros((nb_tangles, n_points), dtype=int)

        for i, tangle in enumerate(tangles):
            cuts = list(tangle.specification.keys())
            orientations = list(tangle.specification.values())

            matching_cuts[i, :] = np.sum((all_cuts[cuts, :].T == orientations), axis=1)

        predictions = np.argmax(matching_cuts, axis=0)
        predictions_by_order[order] = predictions

    return predictions_by_order


def compute_evaluation(ys, predictions):
    evaluation = {}
    evaluation['v_measure_score'] = None
    evaluation['order_max'] = None

    for order, prediction in predictions.items():

        homogeneity, completeness, v_measure_score = \
            homogeneity_completeness_v_measure(ys, prediction)

        if evaluation['v_measure_score'] is None or evaluation['v_measure_score'] < v_measure_score:
            evaluation["homogeneity"] = homogeneity
            evaluation["completeness"] = completeness
            evaluation["v_measure_score"] = v_measure_score
            evaluation['order_max'] = order

    return evaluation


def get_dataset_cuts_order(args):
    if args.verbose >= 2:
        print("Load data\n", flush=True)
    xs, ys, G, order_function = get_dataset_and_order_function(args, args.seed)

    if args.verbose >= 2:
        print("Find cuts", flush=True)
    all_cuts = compute_cuts(xs, args.preprocessing, verbose=args.verbose)

    if args.verbose >= 2:
        print(f"\tI found {len(all_cuts)} cuts\n")
        print("Compute order", flush=True)
    all_cuts, orders = order_cuts(all_cuts, order_function)

    mask_orders_to_pick = orders <= np.percentile(orders, q=args.tangles.percentage_orders)
    orders = orders[mask_orders_to_pick]
    all_cuts = all_cuts[mask_orders_to_pick, :]

    max_considered_order = orders[-1]
    if args.verbose >= 2:
        print(f"\tI will stop at order: {max_considered_order}")
        print(f'\tI will use {len(all_cuts)} cuts\n', flush=True)

    if args.plot.cuts:
        if args.experiment.dataset_type == 'graph':
            plot_cuts(G, all_cuts[:args.plot.nb_cuts], orders, args.experiment.dataset_type, args.output.dir)
        else:
            raise NotImplementedError('I still need to implement this')
    return xs, ys, G, orders, all_cuts


def tangle_computation(args, all_cuts, orders):
    agreement = args.algorithm.agreement

    if args.verbose >= 2:
        print(f"Using agreement = {agreement} \n")
        print("Start tangle computation", flush=True)

    tangles = []
    tangles_of_order = {}

    unique_orders = np.unique(orders)

    for idx_order, order in enumerate(unique_orders):

        idx_cuts_order_i = np.where(np.all([order - 1 < orders, orders <= order], axis=0))[0]

        if len(idx_cuts_order_i) > 0:
            if args.verbose >= 2:
                print(f"\tCompute tangles of order {order}", flush=True)

            cuts_order_i = all_cuts[idx_cuts_order_i]
            tangles = compute_tangles(tangles, cuts_order_i, idx_cuts_order_i,
                                      min_size=agreement, algorithm=args.algorithm)
            if args.verbose >= 2:
                print(f"\t\tI found {len(tangles)} tangles of order {order}", flush=True)

            if not tangles:
                max_considered_order = orders[-1]
                if args.verbose >= 2:
                    print(f'Stopped computation at order {order} instead of {max_considered_order}', flush=True)
                break

            tangles_of_order[order] = deepcopy(tangles)

    return tangles_of_order


def plotting(args, predictions, G, ys, all_cuts):

    if args.verbose >= 2:
        print('Start plotting', flush=True)

    if args.experiment.dataset_type == 'graph':
        plot_heatmap_graph(G=G, all_cuts=all_cuts, predictions=predictions, path=args.output.dir)
    elif args.experiment.dataset_type == 'discrete':
        plot_heatmap(all_cuts=all_cuts, ys=ys, tangles_by_orders=predictions, path=args.output.dir)

    if args.verbose >= 2:
        print('Done plotting', flush=True)
