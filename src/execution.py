import re
from copy import deepcopy

import numpy as np
import pandas as pd

from sklearn.metrics import homogeneity_completeness_v_measure, adjusted_mutual_info_score

from src.config import PREPROCESSING_COARSENING, DATASET_SBM, DATASET_KNN_BLOBS, PREPROCESSING_FID_MAT, \
    PREPROCESSING_SUBMODULAR, PREPROCESSING_BINARIZED_LIKERT
from src.config import PREPROCESSING_USE_FEATURES, PREPROCESSING_KMODES, PREPROCESSING_KARNIG_LIN
from src.config import NAN
from src.preprocessing import find_kmodes_cuts, kernighan_lin, coarsening_cuts, fid_mat, make_submodular, binarize_likert_scale
from src.loading import get_dataset_and_order_function
from src.plotting import plot_graph_cuts, plot_predictions_graph, plot_predictions, plot_cuts
from src.tangles import core_algorithm
from src.utils import change_lower, change_upper


def compute_cuts(data, args, verbose):
    """
    Given a set of points or an adjacency matrix this function returns the set of cuts that we will use
    to compute tangles.
    This different types of preprocessing are available

     1. PREPROCESSING_USE_FEATURES: consider the features as cuts
     2. PREPROCESSING_MAKE_SUBMODULAR: consider the features as cuts and then make them submodular
     3. PREPROCESSING_RANDOM_COVER: Given an adjacency matrix build a cover of the graph and use that as starting
                                          point for creating the cuts

    Parameters
    ----------
    data: dictionary containing all the input data in various representations
    preprocessing: SimpleNamespace
        The parameters of the preprocessing

    Returns
    -------
    cuts: array of shape [n_cuts, n_points]
        The bipartitions that we will use to compute tangles
    """

    cuts_names = None

    if args['experiment']['preprocessing_name'] == PREPROCESSING_USE_FEATURES:
        cuts = (data['xs'] == True).T
    elif args['experiment']['preprocessing_name'] == PREPROCESSING_SUBMODULAR:
        cuts = (data['xs'] == True).T
        cuts = make_submodular(cuts)
    elif args['experiment']['preprocessing_name'] == PREPROCESSING_BINARIZED_LIKERT:
        cuts, cuts_names = binarize_likert_scale(xs=data['xs'],
                                                 range_answers=args['preprocessing']['range_answers'])
    elif args['experiment']['preprocessing_name'] == PREPROCESSING_KARNIG_LIN:
        cuts = kernighan_lin(A=data['A'],
                             nb_cuts=args['preprocessing']['nb_cuts'],
                             lb_f=args['preprocessing']['lb_f'],
                             verbose=verbose)
        cuts = np.unique(cuts, axis=0)
    elif args['experiment']['preprocessing_name'] == PREPROCESSING_KMODES:
        cuts = find_kmodes_cuts(xs=data['xs'],
                                max_nb_clusters=args['preprocessing'][',ax_nb_clusters'])
        cuts = np.unique(cuts, axis=0)
    elif args['experiment']['preprocessing_name'] == PREPROCESSING_COARSENING:
        cuts = coarsening_cuts(A=data['A'],
                               nb_cuts=args['preprocessing']['coarsening.nb_cuts'],
                               n_max=args['preprocessing']['coarsening.n_max'])
        cuts = np.unique(cuts, axis=0)
    elif args['experiment']['preprocessing_name'] == PREPROCESSING_FID_MAT:
        cuts = fid_mat(xs=data['A'],
                       nb_cuts=args['preprocessing']['nb_cuts'],
                       lb_f=args['preprocessing']['lb_f'],
                       verbose=verbose)
        cuts = np.unique(cuts, axis=0)

    return cuts, cuts_names


def order_cuts(cuts, cuts_names, order_function):
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

    cost_cuts = np.zeros(len(cuts), dtype=float)

    for i_cut, cut in enumerate(cuts):
        cost_cuts[i_cut] = order_function(cut)

    idx = np.argsort(cost_cuts)
    if cuts_names is not None:
        cuts_names = cuts_names[idx]

    return cuts[idx], cost_cuts[idx], cuts_names


def compute_clusters(tangles_by_orders, all_cuts, verbose):
    """ 
    
    Given the tangles divided by order and all the cuts it converts the tangles into hard clustering.
    It does this by creating a cluster for each tangle and then assign each point to the tangle 
    for which it satisfies the largest number of constrains.

    Parameters
    ----------
    tangles_by_orders : dict of Orientations
        Dictionary with key the order and value the tangles of that order
    all_cuts : ndarray of shape [nb_cuts]
        All the cuts
    verbose : Int
        The verbosity level

    Returns
    -------
    dict of ndarray
        Dictionary with key the order and value the predicted labels for that order
    """
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
        best_values = np.max(matching_cuts, axis=0)

        nb_best_values = np.sum(matching_cuts == best_values, axis=0)
        predictions[nb_best_values > 1] = NAN

        predictions_by_order[order] = predictions

    return predictions_by_order


def compute_maximal_tangles(tangles_by_orders):
    print("Computing maximal tangles")
    maximals = []
    orders = sorted(tangles_by_orders.keys())

    for order in orders:
        tangles = tangles_by_orders[order]
        new_maximals = tangles.copy()
        for m in maximals:
            for tangle in tangles:
                if all(mspec == tangle.specification[i] for (i, mspec) in m.specification.items()):
                    break
            else:
                new_maximals.append(m)
        maximals = new_maximals

    return maximals


def compute_clusters_maximals(maximal_tangles, all_cuts):
    predictions_by_order = {}

    print(f"\tCompute clusters for maximal tangles", flush=True)
    _, n_points = all_cuts.shape
    nb_tangles = len(maximal_tangles)

    matching_cuts = np.zeros((nb_tangles, n_points), dtype=float)

    for i, tangle in enumerate(maximal_tangles):
        cuts = list(tangle.specification.keys())
        orientations = list(tangle.specification.values())

        matching_cuts[i, :] = np.sum((all_cuts[cuts, :].T == orientations), axis=1) / len(cuts)
    best = np.amax(matching_cuts, axis=0)
    predictions = np.zeros(n_points)
    for p in range(n_points):
        the_best = matching_cuts[:, p] == best[p]
        if the_best.sum() == 1:
            predictions[p] = np.argwhere(the_best)
        else:
            predictions[p] = np.nan
            print(f'Unsure about {p}')

    predictions_by_order[-1] = predictions

    return predictions_by_order


def compute_evaluation(ys, predictions):
    evaluation = {}
    evaluation['ami'] = None
    evaluation['order_best'] = None

    for order, prediction in predictions.items():

        #clustered = prediction != NAN
        
        #homogeneity, completeness, v_measure_score = homogeneity_completeness_v_measure(ys[clustered], prediction[clustered])
        ami = adjusted_mutual_info_score(ys, prediction)

        if evaluation['ami'] is None or evaluation['ami'] < ami:
            evaluation["ami"] = ami
            evaluation['order_best'] = order

    return evaluation


def get_dataset_cuts_order(args):
    if args['verbose'] >= 2:
        print("Load data\n", flush=True)
    data, order_function = get_dataset_and_order_function(args)

    if args['verbose'] >= 2:
        print("Find cuts", flush=True)
    all_cuts, cuts_names = compute_cuts(data, args, verbose=args['verbose'])

    if args['verbose'] >= 2:
        print(f"\tI found {len(all_cuts)} unique cuts\n")
        print("Compute order", flush=True)
    all_cuts, orders, cuts_names = order_cuts(all_cuts, cuts_names, order_function)

    mask_orders_to_pick = orders <= np.percentile(orders, q=args['experiment']['percentile_orders'])
    orders = orders[mask_orders_to_pick]
    all_cuts = all_cuts[mask_orders_to_pick, :]

    max_considered_order = orders[-1]
    if args['verbose'] >= 2:
        print(f"\tI will stop at order: {max_considered_order}")
        print(f'\tI will use {len(all_cuts)} cuts\n', flush=True)

    if args['plot']['cuts']:

        xs = data.get('xs', None)
        ys = data.get('ys', None)
        G = data.get('G', None)

        if G is not None:
            plot_graph_cuts(G, ys, all_cuts[:args['plot']['nb_cuts']], orders, args['plot_dir'])
        if xs is not None:
            plot_cuts(xs, ys, all_cuts[:args['plot']['nb_cuts']], orders, args['plot_dir'])

    return data, orders, all_cuts, cuts_names


def tangle_computation(all_cuts, orders, agreement, verbose):

    if verbose >= 2:
        print(f"Using agreement = {agreement} \n")
        print("Start tangle computation", flush=True)

    tangles = []
    tangles_of_order = {}
    old_order = None

    unique_orders = np.unique(orders)

    for order in unique_orders:

        if old_order is None:
            idx_cuts_order_i = np.where(orders <= order)[0]
        else:
            idx_cuts_order_i = np.where(np.all([orders > old_order, orders <= order], axis=0))[0]

        if len(idx_cuts_order_i) > 0:
            if verbose >= 2:
                print(f"\tCompute tangles of order {order} with {len(idx_cuts_order_i)} new cuts", flush=True)

            cuts_order_i = all_cuts[idx_cuts_order_i]
            tangles = core_algorithm(tangles,
                                     current_cuts=cuts_order_i,
                                     idx_current_cuts=idx_cuts_order_i,
                                     agreement=agreement)
            if verbose >= 2:
                print(f"\t\tI found {len(tangles)} tangles of order {order}", flush=True)

            if not tangles:
                max_considered_order = orders[-1]
                if verbose >= 2:
                    print(f'Stopped computation at order {order} instead of {max_considered_order}', flush=True)
                break

            tangles_of_order[order] = deepcopy(tangles)
        
        old_order = order

    return tangles_of_order


def plotting(data, predictions_by_order, verbose, path):

    path.mkdir(parents=True, exist_ok=True)

    if verbose >= 2:
        print('Start plotting', flush=True)

    xs = data.get('xs', None)
    ys = data.get('ys', None)
    G = data.get('G', None)

    if G is not None:
        plot_predictions_graph(G=G, ys=ys, predictions_of_order=predictions_by_order, path=path)
    if xs is not None:
        plot_predictions(xs=xs, ys=ys, predictions_of_order=predictions_by_order, path=path)
    if verbose >= 2:
        print('Done plotting', flush=True)


def get_parameters(args):

    parameters = {}
    parameters['seed'] = args.seeds

    if args.experiment.dataset_name == DATASET_SBM:
        parameters['block_sizes'] = [args.dataset.sbm.block_sizes]
        parameters['p'] = args.dataset.sbm.ps
        parameters['q'] = args.dataset.sbm.qs
    elif args.experiment.dataset_name == DATASET_KNN_BLOBS:
        parameters['blob_sizes'] = [args.dataset.knn_blobs.blob_sizes]
        parameters['blobs_center'] = args.dataset.knn_blobs.blobs_centers
        parameters['k'] = args.dataset.knn_blobs.ks

    if args.args['experiment']['preprocessing_name'] == PREPROCESSING_KARNIG_LIN:
        parameters['nb_cuts'] = [args.args['preprocessing'].nb_cuts]

    return parameters


def print_tangles_names(name_cuts, tangles_by_order, order_best, verbose, path):

    path.mkdir(parents=True, exist_ok=True)

    if verbose >= 2:
        print(f'Printing answers', flush=True)

    for order, tangles in tangles_by_order.items():

        if len(tangles) > 0:

            questions = list(tangles[0].specification.keys())
            questions_names = name_cuts[questions]

            answers = pd.DataFrame()
            for tangle in tangles:
                tmp = pd.DataFrame([tangle.specification])
                answers = answers.append(tmp)
            #answers = answers.astype(str)

            #useless_columns = (answers.nunique(axis=0) == 1)
            #answers.loc[:, useless_columns] = 'Ignore'

            answers.columns = questions_names

            answers.to_csv(path / f'{order:.2f}.csv', index=False)
            if order == order_best:
                answers.to_csv(path / '..' / 'best.csv', index=False)


def tangles_to_range_answers(tangles, cut_names, interval_values, path):

    # the questions are of the form 'name greater of equal than value'
    # this regex gets the name and the value
    template = re.compile(r"(\w+) .+ (\d+)")

    range_answers = pd.DataFrame()
    for tangle in tangles:

        results = {}
        for cut, orientation in tangle.specification.items():

            name, value = template.findall(cut_names[cut])[0]
            value = int(value)

            old = results.get(name, None)
            if old is None:
                new = interval_values
            else:
                new = old

            if orientation:
                new = change_lower(new, value)
            else:
                new = change_upper(new, value - 1)
            results[name] = new

        range_answers = range_answers.append(pd.DataFrame([results]))

    prettification = lambda i: i if i.left != i.right else i.left
    convert_to_interval = lambda i: pd.Interval(left=i[0], right=i[1], closed='both')

    range_answers = range_answers.applymap(convert_to_interval)
    range_answers = range_answers.reindex(sorted(range_answers.columns), axis=1)

    range_answers.applymap(prettification).to_csv(path / 'range_answers.csv', index=False)

    return range_answers


def centers_in_range_answers(cs, range_answers):

    #name_questions = [f'{q']

    p = []
    for _, row in range_answers.iterrows():
        agreement = []
        for c in cs:
            l = []
            for component_c, answer in zip(c, row.tolist()):
                l.append(component_c in answer)        
            agreement.append(np.mean(l))
        p.append(max(agreement))

    print(range_answers)