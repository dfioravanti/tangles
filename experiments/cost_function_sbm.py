import os
from pathlib import Path
import sys

from scipy.stats import stats

sys.path.append("../src")
import pandas as pd

from sklearn.metrics import normalized_mutual_info_score
from sklearn.neighbors._dist_metrics import DistanceMetric

from src.config import load_validate_parser, set_up_dirs, load_validate_config_file, deactivate_plots
from src.execution import compute_and_save_evaluation, tangle_computation, \
    compute_soft_predictions, compute_hard_predictions, save_time_evaluation, get_data, get_cuts, \
    compute_mindset_prediciton, get_cost_function
from src.baselines import compute_and_save_comparison
from src.my_types import Dataset, Data, Bipartitions
from src.parser import make_parser
from src.plotting import plot_soft_predictions, plot_hard_predictions
from datetime import datetime
import time
import numpy as np

from src.tree_tangles import ContractedTangleTree

PATH = "."

def main(args):
    """
    Main function of the program.
    The execution is divided in the following steps

       1. Load datasets
       2. Find the cuts and compute the costs
       3. For each cut compute the tangles by expanding on the
          previous ones if it is consistent. If its not possible stop
       4. Postprocess in soft and hard clustering

    Parameters
    ----------
    args: SimpleNamespace
       The parameters to the program

    Returns
    -------
    """

    hyperparameters = {**args['experiment'], **args['dataset'], **args['preprocessing'], **args['cut_finding']}
    id_run = datetime.now().strftime("%m-%d")

    if args['verbose'] >= -1:
        print('ID for the run = {}'.format(id_run))
        print('Working with hyperparameters = {}'.format(hyperparameters))
        print('Plot settings = {}'.format(args["plot"]), flush=True)

    if args['runs'] > 1:
        args['verbose'] = 0
        args['plot']['no_plots'] = True
        deactivate_plots(args)

    data = get_block_model(args)

    bipartitions = get_cuts_and_costs(data, args)

    quality = np.zeros_like(bipartitions.costs)

    for i, b in enumerate(bipartitions.values):
        quality[i] = normalized_mutual_info_score(b, data.ys)


    print(stats.spearmanr(bipartitions.costs, quality))

    print(stats.pearsonr(bipartitions.costs, quality))

    result = {'cost': bipartitions.costs, 'quality': quality}

    results = pd.DataFrame(result)
    print(results)

    #results.to_csv(str(PATH + '/cost_mean_sbm.csv'.format(id_run)))

def get_block_model(args):
    p = args['dataset']['p']
    q = args['dataset']['q']
    sizes = args['dataset']['block_sizes']
    ys = np.concatenate([np.ones(sizes[0]), np.zeros(sizes[1])])

    A = np.zeros([sizes[0], sizes[0]])
    B = np.zeros([sizes[0], sizes[0]])
    C = np.zeros([sizes[0], sizes[0]])

    for n in range(sizes[0]):
        for m in range(sizes[0]):
            if n < m:
                r = np.random.rand()
                if r < p:
                    A[n, m] = A[m, n] = 1

                r = np.random.rand()
                if r < p:
                    B[n, m] = B[m, n] = 1

            r = np.random.rand()
            if r < q:
                C[n, m] = 1

    adjacency = np.block([[A, C], [C.transpose(), B]])

    return Data(A=adjacency, ys=ys)

def get_cuts_and_costs(data, args):
    sizes = args['dataset']['block_sizes']
    nb_points = sum(sizes)
    values = []

    for k in range(1, sizes[0]):
        for l in range(sizes[0], nb_points):
            cut = np.zeros(nb_points, dtype=bool)
            cut[:k] = True
            cut[sizes[0]:l] = True
            values.append(cut)

    values = np.array(values)
    costs = np.empty(len(values), dtype=float)

    cost_function = get_cost_function(data, args)

    for i, cut in enumerate(values):
        costs[i] = cost_function(values[i])

    bipartitions = Bipartitions(values=values, costs=costs)

    return bipartitions


if __name__ == '__main__':

    # Make parser and parse command line
    parser = make_parser()
    args_parser = parser.parse_args()
    args = load_validate_parser(args_parser)

    root_dir = Path(__file__).resolve().parent.parent
    if args is None:
        cfg_file_path = root_dir / 'settings.yml'
        args = load_validate_config_file(cfg_file_path)

    args = deactivate_plots(args)
    args = set_up_dirs(args, root_dir=root_dir)

    main(args)
