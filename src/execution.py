import re
from functools import partial

import numpy as np
import pandas as pd
from sklearn.datasets import make_moons, make_blobs
from sklearn.metrics import adjusted_rand_score

from src.types import Dataset, CutFinding, Data, Cuts, Preprocessing, CostFunction
from src.cut_finding import find_kmodes_cuts, kernighan_lin, fid_mat, binning, linear_cuts
from src.loading import make_mindsets, make_likert_questionnaire, load_RETINAL, load_CANCER, load_SBM, load_LFR
from src.plotting import plot_cuts
from src.tangles import core_algorithm
from src.tree_tangles import TangleTree, compute_soft_predictions_children, compute_hard_predictions_node
from src.utils import change_lower, change_upper, normalize
from src.cost_functions import edges_cut_cost, implicit_cost


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

    if args['experiment']['dataset'] == Dataset.breast_cancer_wisconsin:
        xs, ys = load_CANCER(args['dataset']['nb_bins'])

        return Data(xs=xs, ys=ys)

    if args['experiment']['dataset'] == Dataset.SBM:
        A, ys, G = load_SBM(block_sizes=args['dataset']['block_sizes'],
                            p_in=args['dataset']['p'],
                            p_out=args['dataset']['q'],
                            seed=args['experiment']['seed'])

        return Data(ys=ys, A=A, G=G)

    if args['experiment']['dataset'] == Dataset.gaussian_mixture:
        xs, ys = make_blobs(n_samples=args['dataset']['blob_sizes'],
                            centers=args['dataset']['blob_centers'],
                            n_features=args['dataset']['blob_centers'],
                            cluster_std=args['dataset']['blob_variances'],
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


def get_cuts(data, args, verbose):
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

        values = (data.xs == True).T
        return Cuts(values=values)

    if args['experiment']['cut_finding'] == CutFinding.binning:

        values, names = binning(xs=data.xs,
                                range_answers=args['cut_finding']['range_answers'],
                                n_bins=args['cut_finding']['n_bins'])
        return Cuts(values=values, names=names)

    if args['experiment']['cut_finding'] == CutFinding.Kernighan_Lin:

        values = kernighan_lin(A=data.A,
                               nb_cuts=args['cut_finding']['nb_cuts'],
                               lb_f=args['cut_finding']['lb_f'],
                               seed=args['experiment']['seed'],
                               verbose=verbose)
        values = np.unique(values, axis=0)
        return Cuts(values=values)

    if args['experiment']['cut_finding'] == CutFinding.kmodes:

        values = find_kmodes_cuts(xs=data.xs,
                                  max_nb_clusters=args['cut_finding']['max_nb_clusters'])
        values = np.unique(values, axis=0)
        return Cuts(values=values)

    if args['experiment']['cut_finding'] == CutFinding.Fiduccia_Mattheyses:

        values = fid_mat(xs=data.A,
                         nb_cuts=args['cut_finding']['nb_cuts'],
                         lb_f=args['cut_finding']['lb_f'],
                         seed=args['experiment']['seed'],
                         verbose=verbose)
        values = np.unique(values, axis=0)
        return Cuts(values=values)

    if args['experiment']['cut_finding'] == CutFinding.linear:

        values, equations = linear_cuts(xs=data.xs,
                                        equations=args['cut_finding']['equations'],
                                        verbose=verbose)

        return Cuts(values=values, equations=equations)

    raise ValueError('Wrong name for a cut finding function')


def apply_preprocess(data, args):

    if args['experiment']['preprocessing'] == Preprocessing.none:
        return data

    if args['experiment']['preprocessing'] == Preprocessing.feature_map:
        raise NotImplementedError('TODO')

    if args['experiment']['preprocessing'] == Preprocessing.knn_graph:
        raise NotImplementedError('TODO')

    if args['experiment']['preprocessing'] == Preprocessing.radius_neighbors_graph:
        raise NotImplementedError('TODO')

    raise ValueError('Wrong name for a preprocessing function')


def get_cost_function(data, args):

    if args['experiment']['cost_function'] == CostFunction.implicit:
        if data.xs is None:
            raise ValueError('You need xs to compute the implicit cost function')

        return partial(implicit_cost, data.xs, args['cost_function']['nb_points'])

    if args['experiment']['cost_function'] == CostFunction.nb_edges_cut:
        if data.A is None:
            raise ValueError('You need A to compute the edge cost')

        return partial(edges_cut_cost, data.A)

    raise ValueError('Wrong name for a cost function')


def compute_cost_and_order_cuts(cuts, cost_function):
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

    cost_cuts = np.zeros(len(cuts.values), dtype=float)
    for i_cut, cut in enumerate(cuts.values):
        cost_cuts[i_cut] = cost_function(cut)
    idx = np.argsort(cost_cuts)

    cuts.values = cuts.values[idx]
    cuts.costs = cost_cuts[idx]
    if cuts.names is not None:
        cuts.names = cuts.names[idx]
    if cuts.equations is not None:
        cuts.equations = cuts.equations[idx]

    return cuts


def pick_cuts_up_to_order(cuts, percentile):
    """
    Drop the cuts whose order is in a percentile above percentile.

    Parameters
    ----------
    cuts: Cuts
    percentile

    Returns
    -------
    """

    mask_orders_to_pick = cuts.costs <= np.percentile(cuts.costs, q=percentile)
    cuts.costs = cuts.costs[mask_orders_to_pick]
    cuts.values = cuts.values[mask_orders_to_pick, :]
    if cuts.names is not None:
        cuts.names = cuts.names[mask_orders_to_pick]
    if cuts.equations is not None:
        cuts.equations = cuts.equations[mask_orders_to_pick]

    return cuts


def get_data_and_cuts(args):
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
        print("Load data\n", flush=True)
    data = get_dataset(args)

    if args['verbose'] >= 2:
        print("Find cuts", flush=True)
    cuts = get_cuts(data, args, verbose=args['verbose'])
    if args['verbose'] >= 2:
        print(f'\tI found {len(cuts.values)} cuts\n')

    print("Compute cost", flush=True)
    cost_function = get_cost_function(data, args)
    cuts = compute_cost_and_order_cuts(cuts, cost_function)

    cuts = pick_cuts_up_to_order(cuts,
                                 percentile=args['experiment']['percentile_orders'])
    if args['verbose'] >= 2:
        max_considered_order = cuts.costs[-1]
        print(f"\tI will stop at order: {max_considered_order}")
        print(f'\tI will use {len(cuts.values)} cuts\n', flush=True)

    if args['plot']['cuts']:
        if args['verbose'] >= 2:
            print(f"\tPlotting cuts")

        plot_cuts(data, cuts,
                  nb_cuts_to_plot=args['plot']['nb_cuts'],
                  path=args['plot_dir'])

    return data, cuts


def tangle_computation(cuts, agreement, verbose):
    """

    Parameters
    ----------
    cuts: Cuts
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
        print(f"Using agreement = {agreement} \n")
        print("Start tangle computation", flush=True)

    tangles_tree = TangleTree()
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
                print(f"\tCompute tangles of order {order} with {len(idx_cuts_order_i)} new cuts", flush=True)

            cuts_order_i = cuts.values[idx_cuts_order_i]
            new_tree = core_algorithm(tangles_tree=tangles_tree,
                                      current_cuts=cuts_order_i,
                                      idx_current_cuts=idx_cuts_order_i,
                                      agreement=agreement)

            if new_tree is None:
                max_order = cuts.costs[-1]
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
            # answers = answers.astype(str)

            # useless_columns = (answers.nunique(axis=0) == 1)
            # answers.loc[:, useless_columns] = 'Ignore'

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


def compute_soft_predictions(contracted_tree, cuts, verbose):
    costs = np.exp(-normalize(cuts.costs))

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
    _, nb_points = cuts.values.shape
    idx_points = np.arange(nb_points)
    ys_predicted = np.zeros(nb_points, dtype=int)

    if not condensed_tree.is_empty:

        clusters = compute_hard_predictions_node(node=condensed_tree.root,
                                                 idx_points=idx_points,
                                                 max_tangles=condensed_tree.maximals)

        for y, idx_points in clusters.items():
            ys_predicted[idx_points] = y

    return ys_predicted
