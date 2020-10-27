import csv
import multiprocessing
import re
from functools import partial

import numpy as np
import os
import pandas as pd
from sklearn.cluster import SpectralClustering, KMeans, AgglomerativeClustering
from sklearn.datasets import make_moons, make_blobs, make_circles
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors._dist_metrics import DistanceMetric

from src.my_types import Dataset, CutFinding, Data, Cuts, Preprocessing, CostFunction
from src.cut_finding import kernighan_lin, fid_mat, binning, linear_cuts, \
    random_projection_mean
from src.loading import make_mindsets, make_likert_questionnaire, load_RETINAL, load_CANCER, load_SBM, load_LFR
from src.plotting import plot_cuts
from src.preprocessing import calculate_knn_graph, calculate_radius_graph, calculate_weighted_knn_graph
from src.tangles import core_algorithm
from src.tree_tangles import TangleTree, compute_soft_predictions_children
from src.utils import normalize
from src.cost_functions import edges_cut_cost, mean_euclidean_distance, euclidean_distance, \
    manhattan_distance, mean_manhattan_distance, mean_edges_cut_cost
import time


def get_dataset(args):
    """
    Function that returns the desired dataset and the order function in the format that we expect.
    Datasets are always in the format of
        - xs: Features that we need for clustering, like questions for the questionnaire or the adjacency matrix for
              the graph
        - ys: Class label
    Order functions are assumed to be functions that only need a bipartition as inputs and return the order
    of that bipartion. We assume that all the other args['dataset'] are loaded via partial evaluation in this function.

    args['dataset']
    ----------
    dataset: SimpleNamespace
        The args['dataset'] of the dataset to load
    seed: int
        The seed for the RNG

    Returns
    -------
    data: Data
        the dataset in various representation
    """

    if args['experiment']['dataset'] == Dataset.mindsets:
        xs, ys, cs = make_mindsets(mindset_sizes=args['dataset']['mindset_sizes'],
                                   nb_questions=args['dataset']['nb_questions'],
                                   nb_useless=args['dataset']['nb_useless'],
                                   noise=args['dataset']['noise'],
                                   seed=args['experiment']['seed'])

        return Data(xs=xs, ys=ys, cs=cs)

    if args['experiment']['dataset'] == Dataset.questionnaire_likert:
        xs, ys, cs = make_likert_questionnaire(nb_samples=args['dataset']['nb_samples'],
                                               nb_features=args['dataset']['nb_features'],
                                               nb_mindsets=args['dataset']['nb_mindsets'],
                                               centers=args['dataset']['centers'],
                                               range_answers=args['dataset']['range_answers'],
                                               seed=args['experiment']['seed'])

        return Data(xs=xs, ys=ys, cs=cs)

    if args['experiment']['dataset'] == Dataset.retinal:
        xs, ys = load_RETINAL(root_path=args['root_dir'],
                              nb_bins=args['dataset']['nb_bins'],
                              max_idx=args['dataset']['max_idx'])

        return Data(xs=xs, ys=ys)

    if args['experiment']['dataset'] == Dataset.moons:
        xs, ys = make_moons(n_samples=args['dataset']['n_samples'],
                            noise=args['dataset']['noise'],
                            random_state=args['experiment']['seed'])

        return Data(xs=xs, ys=ys)

    if args['experiment']['dataset'] == Dataset.circles:
        xs, ys = make_circles(n_samples=args['dataset']['n_samples'],
                              factor=args['dataset']['factor'],
                              noise=args['dataset']['noise'])

        return Data(xs=xs, ys=ys)

    if args['experiment']['dataset'] == Dataset.breast_cancer_wisconsin:
        xs, ys = load_CANCER()

        return Data(xs=xs, ys=ys)

    if args['experiment']['dataset'] == Dataset.SBM:
        A, ys, G = load_SBM(block_sizes=args['dataset']['block_sizes'],
                            p_in=args['dataset']['p'],
                            p_out=args['dataset']['q'],
                            seed=args['experiment']['seed'])

        return Data(ys=ys, A=A, G=G)

    if args['experiment']['dataset'] == Dataset.gaussian_mixture:
        xs, ys = make_blobs(n_samples=args['dataset']['sizes'],
                            centers=args['dataset']['centers'],
                            n_features=len(args['dataset']['centers']),
                            cluster_std=args['dataset']['variances'],
                            random_state=args['experiment']['seed'])

        return Data(xs=xs, ys=ys)

    if args['experiment']['dataset'] == Dataset.LFR:
        A, ys, G = load_LFR(nb_nodes=args['dataset']['nb_nodes'],
                            tau1=args['dataset']['tau1'],
                            tau2=args['dataset']['tau2'],
                            mu=args['dataset']['mu'],
                            average_degree=args['dataset']['average_degree'],
                            min_community=args['dataset']['min_community'],
                            seed=args['experiment']['seed'])

        return Data(ys=ys, A=A, G=G)

    if args['experiment']['dataset'] == Dataset.wave:
        df = pd.read_csv('datasets/waveform.csv')
        xs = df[df.columns[:-1]].to_numpy()
        ys = df[df.columns[-1]].to_numpy()

        return Data(xs=xs, ys=ys)

    raise ValueError('Wrong name for a dataset')


def find_bipartitions(data, args, verbose):
    """
    Given a set of points or an adjacency matrix this function returns the set of cuts that we will use
    to compute tangles. If it makes sense it computes the names and equations of the cuts for better interpretability
    and post-processing.

    Parameters
    ----------
    data: Data
        all the input data in various representations
    args: SimpleNamespace
        The arguments to the program
    verbose: int
        the verbose level of the printing

    Returns
    -------
    cuts: Cuts
        the cuts that we will use
    """

    if args['experiment']['cut_finding'] == CutFinding.features:
        start = time.time()
        values = (data.xs == True).T
        end = time.time() - start

        return Cuts(values=values), end

    if args['experiment']['cut_finding'] == CutFinding.binning:
        start = time.time()
        values, names = binning(xs=data.xs,
                                n_bins=args['cut_finding']['n_bins'])
        end = time.time() - start

        return Cuts(values=values, names=names), end

    if args['experiment']['cut_finding'] == CutFinding.Kernighan_Lin:
        start = time.time()
        values = kernighan_lin(A=data.A,
                               nb_cuts=args['cut_finding']['nb_cuts'],
                               lb_f=args['cut_finding']['lb_f'],
                               seed=args['experiment']['seed'],
                               verbose=verbose,
                               early_stopping=args['cut_finding']['early_stopping'])
        end = time.time() - start

        values = np.unique(values, axis=0)
        return Cuts(values=values), end

    # if args['experiment']['cut_finding'] == CutFinding.kmodes:
    #
    #     values = find_kmodes_cuts(xs=data.xs,
    #                               max_nb_clusters=args['cut_finding']['max_nb_clusters'])
    #     values = np.unique(values, axis=0)
    #     return Cuts(values=values)

    if args['experiment']['cut_finding'] == CutFinding.Fiduccia_Mattheyses:
        start = time.time()
        values = fid_mat(xs=data.A,
                         nb_cuts=args['cut_finding']['nb_cuts'],
                         lb_f=args['cut_finding']['lb_f'],
                         seed=args['experiment']['seed'],
                         verbose=verbose,
                         early_stopping=args['cut_finding']['early_stopping'])
        values = np.unique(values, axis=0)
        end = time.time() - start
        return Cuts(values=values), end

    if args['experiment']['cut_finding'] == CutFinding.linear:
        start = time.time()
        values, equations = linear_cuts(xs=data.xs,
                                        equations=args['cut_finding']['equations'],
                                        verbose=verbose)

        end = time.time() - start
        return Cuts(values=values, equations=equations), end

    if args['experiment']['cut_finding'] == CutFinding.random_projection:
        start = time.time()
        values = random_projection_mean(xs=data.xs,
                                        nb_cuts=args['cut_finding']['nb_cuts'],
                                        seed=args['experiment']['seed'])
        end = time.time() - start
        return Cuts(values=values), end

    raise ValueError('Wrong name for a cut finding function')


def apply_preprocess(data, args):
    if args['experiment']['preprocessing'] == Preprocessing.none:
        return data

    if args['experiment']['preprocessing'] == Preprocessing.knn_graph:
        return calculate_knn_graph(data, args['preprocessing']['k'])

    if args['experiment']['preprocessing'] == Preprocessing.radius_neighbors_graph:
        return calculate_radius_graph(data, args['preprocessing']['radius'])

    if args['experiment']['preprocessing'] == Preprocessing.weighted_knn_graph:
        return calculate_weighted_knn_graph(data, args['preprocessing']['k'])

    raise ValueError('Wrong name for a preprocessing function')


def get_cost_function(data, args):
    if args['experiment']['cost_function'] == CostFunction.euclidean:
        if data.xs is None:
            raise ValueError('You need xs to compute the euclidean cost function')

        return partial(euclidean_distance, data.xs, args['experiment']['nb_sample_points'])

    if args['experiment']['cost_function'] == CostFunction.mean_euclidean:
        if data.xs is None:
            raise ValueError('You need xs to compute the euclidean cost function')

        return partial(mean_euclidean_distance, data.xs, args['experiment']['nb_sample_points'])

    if args['experiment']['cost_function'] == CostFunction.manhattan:
        if data.xs is None:
            raise ValueError('You need xs to compute the manhattan cost function')

        return partial(manhattan_distance, data.xs, args['experiment']['nb_sample_points'])

    if args['experiment']['cost_function'] == CostFunction.mean_manhattan:
        if data.xs is None:
            raise ValueError('You need xs to compute the manhattan cost function')

        return partial(mean_manhattan_distance, data.xs, args['experiment']['nb_sample_points'])

    if args['experiment']['cost_function'] == CostFunction.cut_value:
        if data.A is None:
            raise ValueError('You need a graph to compute the edge cost')

        return partial(edges_cut_cost, data.A, args['experiment']['nb_sample_points'])

    if args['experiment']['cost_function'] == CostFunction.mean_cut_value:
        if data.A is None:
            raise ValueError('You need a graph to compute the edge cost')

        return partial(mean_edges_cut_cost, data.A, args['experiment']['nb_sample_points'])

    raise ValueError('Wrong name for a cost function')


def compute_cost_and_order_cuts(bipartitions, cost_function):
    """
    Compute the cost of a series of cuts and costs them according to their cost

    Parameters
    ----------
    cuts: Cuts
        the cuts that we will consider
    cost_function: function
        The order function

    Returns
    -------
    cuts: Cuts
        the cuts ordered by costs
    """

    cost_bipartitions = np.zeros(len(bipartitions.values), dtype=float)
    for i_cut, cut in enumerate(bipartitions.values):
        cost_bipartitions[i_cut] = cost_function(cut)

    idx = np.argsort(cost_bipartitions)

    bipartitions.values = bipartitions.values[idx]
    bipartitions.costs = cost_bipartitions[idx]
    if bipartitions.names is not None:
        bipartitions.names = bipartitions.names[idx]
    if bipartitions.equations is not None:
        bipartitions.equations = bipartitions.equations[idx]

    bipartitions.order = np.argsort(idx)

    return bipartitions


def pick_cuts_up_to_order(bipartitions, percentile):
    """
    Drop the cuts whose order is in a percentile above percentile.

    Parameters
    ----------
    cuts: Cuts
    percentile

    Returns
    -------
    """
    mask_orders_to_pick = bipartitions.costs <= np.percentile(bipartitions.costs[~np.isnan(bipartitions.costs)],
                                                              q=percentile)
    bipartitions.costs = bipartitions.costs[mask_orders_to_pick]
    bipartitions.values = bipartitions.values[mask_orders_to_pick, :]
    if bipartitions.names is not None:
        bipartitions.names = bipartitions.names[mask_orders_to_pick]
    if bipartitions.equations is not None:
        bipartitions.equations = bipartitions.equations[mask_orders_to_pick]

    return bipartitions


def get_data(args):
    if args['verbose'] >= 2:
        print("Load data\n", flush=True)
    data = get_dataset(args)

    if args['verbose'] >= 2:
        print("Preprocessing data\n", flush=True)
    data = apply_preprocess(data, args)

    return data


def get_cuts(args, data):
    """
    Function to load the datasets, compute the cuts and the costs.

    Parameters
    ----------
    args: SimpleNamespace
        The arguments to the program

    Returns
    -------
    data: Data
    cuts: Cuts
    """

    if args['verbose'] >= 2:
        print("Find cuts", flush=True)

    bipartitions, bipartitions_time = find_bipartitions(data, args, verbose=args['verbose'])

    if args['verbose'] >= 2:
        print('\tI found {} cuts\n'.format(len(bipartitions.values)))

    print("Compute cost", flush=True)
    cost_function = get_cost_function(data, args)
    start = time.time()
    bipartitions = compute_cost_and_order_cuts(bipartitions, cost_function)
    cost_and_sort_time = time.time() - start

    bipartitions = pick_cuts_up_to_order(bipartitions,
                                         percentile=args['experiment']['percentile_orders'])
    if args['verbose'] >= 2:
        max_considered_order = bipartitions.costs[-1]
        print("\tI will stop at order: {}".format(max_considered_order))
        print('\tI will use {} cuts\n'.format(len(bipartitions.values)), flush=True)

    if args['plot']['cuts']:
        if args['verbose'] >= 2:
            print("\tPlotting cuts")

        plot_cuts(data, bipartitions,
                  nb_cuts_to_plot=args['plot']['nb_cuts'],
                  path=args['plot_dir'])

    return bipartitions, bipartitions_time, cost_and_sort_time


def tangle_computation(cuts, agreement, verbose):
    """

    Parameters
    ----------
    bipartitions: bipartitions
    agreement: int
        The agreement parameter
    verbose:
        verbosity level
    Returns
    -------
    tangles_tree: TangleTree
        The tangle search tree
    """

    if verbose >= 2:
        print("Using agreement = {} \n".format(agreement))
        print("Start tangle computation", flush=True)

    tangles_tree = TangleTree(agreement=agreement)
    old_order = None

    unique_orders = np.unique(cuts.costs)

    for order in unique_orders:

        if old_order is None:
            idx_cuts_order_i = np.where(cuts.costs <= order)[0]
        else:
            idx_cuts_order_i = np.where(np.all([cuts.costs > old_order,
                                                cuts.costs <= order], axis=0))[0]

        if len(idx_cuts_order_i) > 0:

            if verbose >= 2:
                print("\tCompute tangles of order {} with {} new cuts".format(order, len(idx_cuts_order_i)), flush=True)

            cuts_order_i = cuts.values[idx_cuts_order_i]
            new_tree = core_algorithm(tree=tangles_tree,
                                      current_cuts=cuts_order_i,
                                      idx_current_cuts=idx_cuts_order_i)

            if new_tree is None:
                max_order = cuts.costs[-1]
                if verbose >= 2:
                    print('\t\tI could not add all the new cuts')
                    print('\n\tI stopped the computation at order {} instead of {}'.format(old_order, max_order),
                          flush=True)
                break
            else:
                tangles_tree = new_tree

                if verbose >= 2:
                    print("\t\tI found {} tangles of order less or equal {}".format(len(new.active), order),
                          flush=True)

        old_order = order

    if tangles_tree is not None:
        tangles_tree.maximals += tangles_tree.active

    return tangles_tree


def print_tangles_names(name_cuts, tangles_by_order, order_best, verbose, path):
    path.mkdir(parents=True, exist_ok=True)

    if verbose >= 2:
        print('Printing answers', flush=True)

    for order, tangles in tangles_by_order.items():

        if len(tangles) > 0:

            questions = list(tangles[0].specification.keys())
            questions_names = name_cuts[questions]

            answers = pd.DataFrame()
            for tangle in tangles:
                tmp = pd.DataFrame([tangle.specification])
                answers = answers.append(tmp)
            # answers = answers.astype(str)

            # useless_columns = (answers.nunique(axis=0) == 1)
            # answers.loc[:, useless_columns] = 'Ignore'

            answers.columns = questions_names

            answers.to_csv(path / '{}.csv'.format(np.round(order, 2)), index=False)
            if order == order_best:
                answers.to_csv(path / '..' / 'best.csv', index=False)


# def tangles_to_range_answers(tangles, cut_names, interval_values, path):
#     # the questions are of the form 'name greater or equal than value'
#     # this regex gets the name and the value
#     template = re.compile(r"(\w+) .+ (\d+)")
#
#     range_answers = pd.DataFrame()
#     for tangle in tangles:
#
#         results = {}
#         for cut, orientation in tangle.specification.items():
#
#             name, value = template.findall(cut_names[cut])[0]
#             value = int(value)
#
#             old = results.get(name, None)
#             if old is None:
#                 new = interval_values
#             else:
#                 new = old
#
#             if orientation:
#                 new = change_lower(new, value)
#             else:
#                 new = change_upper(new, value - 1)
#             results[name] = new
#
#         range_answers = range_answers.append(pd.DataFrame([results]))
#
#     prettification = lambda i: i if i.left != i.right else i.left
#     convert_to_interval = lambda i: pd.Interval(left=i[0], right=i[1], closed='both')
#
#     range_answers = range_answers.applymap(convert_to_interval)
#     range_answers = range_answers.reindex(sorted(range_answers.columns), axis=1)
#
#     range_answers.applymap(prettification).to_csv(path / 'range_answers.csv', index=False)
#
#     return range_answers


def clean_str(element):
    string = str(element)
    string = string.replace(" ", "")
    return string


def centers_in_range_answers(cs, range_answers):
    # name_questions = [f'{q']

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


def compute_soft_predictions(contracted_tree, cuts, verbose=0):
    costs = np.exp(-normalize(cuts.costs))

    compute_soft_predictions_children(node=contracted_tree.root,
                                      cuts=cuts,
                                      costs=costs,
                                      verbose=verbose)

    contracted_tree.processed_soft_prediction = True


def compute_mindset_prediciton(xs, cs):
    metric = DistanceMetric.get_metric('manhattan')

    distance = metric.pairwise(xs, cs)
    predicted = np.empty(xs.shape[0])

    for i, d in enumerate(distance):
        predicted[i] = np.random.choice(np.flatnonzero(d == d.min()))

    return predicted


def compute_and_save_evaluation(ys, ys_predicted, hyperparameters, id_run, path, r=1):
    ARS = adjusted_rand_score(ys, ys_predicted)
    NMI = normalized_mutual_info_score(ys, ys_predicted)

    print('Found {} clusters.'.format(len(np.unique(ys_predicted))))
    print('Adjusted Rand Score: {}'.format(ARS), flush=True)
    print('Normalized Mutual Information: {}'.format(NMI), flush=True)

    results = pd.Series({**hyperparameters}).to_frame().T
    results['Adjusted Rand Score'] = ARS
    results['Normalized Mutual Information'] = NMI
    results['number found clusters'] = len(np.unique(ys_predicted))

    results.index = range(r, r + 1)

    if os.path.isfile(str(path / 'evaluation_{}.csv'.format(id_run))):
        results.to_csv(str(path / 'evaluation_{}.csv'.format(id_run)), mode='a', header=False)
    else:
        results.to_csv(str(path / 'evaluation_{}.csv'.format(id_run)))


def save_time_evaluation(id_run, pre_time, cost_time, tst_time, post_time, all_time, path, verbose, r):
    field_names = ["", "all", "bipartitions", "calculate cost and order", "build tangle search tree", "soft clustering"]
    results = [{"": r, "all": all_time, "bipartitions": pre_time, "calculate cost and order": cost_time,
                "build tangle search tree": tst_time, "soft clustering": post_time}]

    if verbose > 1:
        print("runtimes: ", results[0]['all'])

    first = not os.path.isfile(str(path / 'runtime_{}.csv'.format(id_run)))

    with open(str(path / 'runtime_{}.csv'.format(id_run)), 'a') as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        if first:
            writer.writeheader()
        writer.writerows(results)


def compute_hard_predictions(condensed_tree, cuts, xs=None):
    if xs is not None:
        cs = []
        nb_cuts = len(cuts.values)

        for leaf in condensed_tree.maximals:
            c = np.full(nb_cuts, 0.5)
            tangle = leaf.tangle
            c[list(tangle.specification.keys())] = np.array(list(tangle.specification.values()), dtype=int)
            cs.append(c[cuts.order])

        cs = np.array(cs)

        return compute_mindset_prediciton(xs, cs), cs

    else:
        if not condensed_tree.processed_soft_prediction:
            print("No probabilities given yet. Calculating soft predictions first!")
            compute_soft_predictions(condensed_tree, cuts)

        probabilities = []
        for node in condensed_tree.maximals:
            probabilities.append(node.p)

        ys_predicted = np.argmax(probabilities, axis=0)

        return ys_predicted, None
