import re
from copy import deepcopy

import numpy as np
import pandas as pd

from sklearn.metrics import adjusted_rand_score

from src.config import PREPROCESSING_COARSENING, DATASET_SBM, DATASET_BLOBS, PREPROCESSING_FID_MAT, \
    PREPROCESSING_SUBMODULAR, PREPROCESSING_BINARIZED_LIKERT, PREPROCESSING_LINEAR_CUTS
from src.config import PREPROCESSING_USE_FEATURES, PREPROCESSING_KMODES, PREPROCESSING_KARNIG_LIN
from src.config import NAN
from src.preprocessing import find_kmodes_cuts, kernighan_lin, coarsening_cuts, fid_mat, \
                              binarize_likert_scale, linear_cuts
from src.loading import get_dataset_and_order_function
from src.plotting import plot_graph_cuts, plot_cuts
from src.tangles import core_algorithm
from src.tree_tangles import TangleTree, compute_soft_predictions_children, compute_hard_predictions_node
from src.utils import change_lower, change_upper, normalize

import matplotlib.pyplot as plt


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

    cuts = {}
    cuts['names'] = None
    cuts['equations'] = None

    if args['experiment']['preprocessing_name'] == PREPROCESSING_USE_FEATURES:
        
        cuts['values'] = (data['xs'] == True).T
        
    elif args['experiment']['preprocessing_name'] == PREPROCESSING_SUBMODULAR:
        
        cuts['values'] = make_submodular((data['xs'] == True).T)
        
    elif args['experiment']['preprocessing_name'] == PREPROCESSING_BINARIZED_LIKERT:
        
        sets, names = binarize_likert_scale(xs=data['xs'],
                                            range_answers=args['preprocessing']['range_answers'])
        cuts['values'] = sets
        cuts['names'] = names
        
    elif args['experiment']['preprocessing_name'] == PREPROCESSING_KARNIG_LIN:
        
        sets = kernighan_lin(A=data['A'],
                             nb_cuts=args['preprocessing']['nb_cuts'],
                             lb_f=args['preprocessing']['lb_f'],
                             seed=args['experiment']['seed'],
                             verbose=verbose)
        sets = np.unique(sets, axis=0)
        cuts['values'] = sets
        
    elif args['experiment']['preprocessing_name'] == PREPROCESSING_KMODES:
        
        sets = find_kmodes_cuts(xs=data['xs'],
                                max_nb_clusters=args['preprocessing'][',ax_nb_clusters'])
        sets = np.unique(sets, axis=0)
        cuts['values'] = sets
        
    elif args['experiment']['preprocessing_name'] == PREPROCESSING_COARSENING:
        
        sets = coarsening_cuts(A=data['A'],
                               nb_cuts=args['preprocessing']['coarsening.nb_cuts'],
                               n_max=args['preprocessing']['coarsening.n_max'])
        sets = np.unique(sets, axis=0)
        cuts['values'] = sets
        
    elif args['experiment']['preprocessing_name'] == PREPROCESSING_FID_MAT:
        
        sets = fid_mat(xs=data['A'],
                       nb_cuts=args['preprocessing']['nb_cuts'],
                       lb_f=args['preprocessing']['lb_f'],
                       seed=args['experiment']['seed'],
                       verbose=verbose)
        sets = np.unique(sets, axis=0)
        cuts['values'] = sets
        
    elif args['experiment']['preprocessing_name'] == PREPROCESSING_LINEAR_CUTS:
        
        sets, equations = linear_cuts(xs=data['xs'],
                                 equations=args['preprocessing']['equations'],
                                 verbose=verbose)

        cuts['values'] = sets
        cuts['equations'] = equations
        
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

    value_cuts, name_cuts, eq_cuts = cuts['values'], cuts['names'], cuts['equations']

    cost_cuts = np.zeros(len(value_cuts), dtype=float)

    for i_cut, cut in enumerate(value_cuts):
        cost_cuts[i_cut] = order_function(cut)

    idx = np.argsort(cost_cuts)
    if name_cuts is not None:
        name_cuts = name_cuts[idx]
        
    if eq_cuts is not None:
        eq_cuts = eq_cuts[idx]

    return cuts, cost_cuts[idx]


def pick_cuts_up_to_order(cuts, orders, percentile):
    
    # TODO: Remove names and eq too but it still works now
    
    mask_orders_to_pick = orders <= np.percentile(orders, q=percentile)
    orders = orders[mask_orders_to_pick]
    cuts['values'] = cuts['values'][mask_orders_to_pick, :]

    return cuts, orders


def get_dataset_cuts_order(args):
    if args['verbose'] >= 2:
        print("Load data\n", flush=True)
    data, order_function = get_dataset_and_order_function(args)

    if args['verbose'] >= 2:
        print("Find cuts", flush=True)
    cuts = compute_cuts(data, args, verbose=args['verbose'])

    if args['verbose'] >= 2:
        print(f"\tI found {len(cuts)} unique cuts\n")
        print("Compute order", flush=True)
    cuts, orders = order_cuts(cuts, order_function)

    cuts, orders = pick_cuts_up_to_order(cuts, orders, percentile=args['experiment']['percentile_orders'])
    max_considered_order = orders[-1]
    if args['verbose'] >= 2:
        print(f"\tI will stop at order: {max_considered_order}")
        print(f'\tI will use {len(cuts["values"])} cuts\n', flush=True)

    if args['plot']['cuts']:
        if args['verbose'] >= 2:
            print(f"\tPlotting cuts")
            
        plot_cuts(data, cuts, orders, 
                  nb_cuts_to_plot=args['plot']['nb_cuts'], 
                  path=args['plot_dir'])
        
    return data, orders, cuts


def tangle_computation(cuts, orders, agreement, verbose):

    if verbose >= 2:
        print(f"Using agreement = {agreement} \n")
        print("Start tangle computation", flush=True)

    tangles_tree = TangleTree()
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

            cuts_order_i = cuts['values'][idx_cuts_order_i]
            new_tree = core_algorithm(tangles_tree=tangles_tree,
                                      current_cuts=cuts_order_i,
                                      idx_current_cuts=idx_cuts_order_i,
                                      agreement=agreement)

            if new_tree is None:
                max_order = orders[-1]
                if verbose >= 2:
                    print('\t\tI could not add all the new cuts')
                    print(f'\n\tI stopped the computation at order {old_order} instead of {max_order}', flush=True)
                break
            else:
                tangles_tree = new_tree

                if verbose >= 2:
                    print(f"\t\tI found {len(new_tree.active)} tangles of order {order}", flush=True)
        
        old_order = order

    if tangles_tree is not None:
        tangles_tree.maximals += tangles_tree.active

    return tangles_tree


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


def compute_soft_predictions(contracted_tree, cuts, orders, verbose):

    costs = np.exp(-normalize(orders))

    compute_soft_predictions_children(node=contracted_tree.root,
                                  cuts=cuts,
                                  costs=costs,
                                  verbose=verbose)

    
def compute_and_save_evaluation(ys, ys_predicted, hyperparameters, id_run, path):

    ARS = adjusted_rand_score(ys, ys_predicted)

    print(f'Adjusted Rand Score: {ARS}', flush=True)

    results = pd.Series({**hyperparameters}).to_frame().T
    results['Adjusted Rand Score'] = ARS

    results.to_csv(path / f'evaluation_{id_run}.csv')


def compute_hard_preditions(condensed_tree, cuts):
    
    _, nb_points = cuts.shape

    idx_points = np.arange(nb_points)
    ys_predicted = np.zeros(nb_points, dtype=int)

    clusters = compute_hard_predictions_node(node=condensed_tree.root,
                                             idx_points=idx_points,
                                             max_tangles=condensed_tree.maximals)

    for y, idx_points in clusters.items():
        ys_predicted[idx_points] = y

    return ys_predicted