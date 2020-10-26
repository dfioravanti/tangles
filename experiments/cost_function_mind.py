import os
from pathlib import Path
import sys

sys.path.append("../src")
import pandas as pd

from sklearn.metrics import normalized_mutual_info_score
from sklearn.neighbors._dist_metrics import DistanceMetric

from src.config import load_validate_parser, set_up_dirs, load_validate_config_file, deactivate_plots
from src.execution import compute_and_save_evaluation, tangle_computation, \
    compute_soft_predictions, compute_hard_predictions, save_time_evaluation, get_data, get_cuts, \
    compute_mindset_prediciton
from src.baselines import compute_and_save_comparison
from src.my_types import Dataset, Data
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

    data = get_mindset_data()

    bipartitions, _, _ = get_cuts(args, data)

    bipartitions.values = bipartitions.values[bipartitions.order]
    bipartitions.costs = bipartitions.costs[bipartitions.order]

    quality = np.zeros_like(bipartitions.costs)

    for i, b in enumerate(bipartitions.values):
        quality[i] = normalized_mutual_info_score(b, data.ys)

    result = {'cost': bipartitions.costs, 'quality': quality}

    results = pd.DataFrame(result)
    results.to_csv(str(PATH + '/cost_sum.csv'.format(id_run)))


def get_mindset_data():
    mindset_sizes = [200, 200]
    nb_questions = 10
    nb_useless = 5
    noise = 0.1

    nb_points = sum(mindset_sizes)
    nb_mindsets = len(mindset_sizes)

    xs, ys = [], []

    # create ground truth mindset
    mindsets = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

    for idx_mindset, size_mindset in enumerate(mindset_sizes):

        # Points without noise
        xs_mindset = np.tile(mindsets[idx_mindset], (size_mindset, 1))
        ys_mindset = np.repeat(idx_mindset, repeats=size_mindset, axis=0)

        xs.append(xs_mindset)
        ys.append(ys_mindset)

    xs = np.vstack(xs)
    ys = np.concatenate(ys)

    # Add noise
    noise_per_question = np.random.rand(nb_points, nb_questions)
    flip_question = noise_per_question < noise
    xs[flip_question] = np.logical_not(xs[flip_question])

    # add noise question like gender etc.
    if nb_useless is not None:
        mindsets = np.hstack((mindsets, np.full([nb_mindsets, nb_useless], 0.5)))
        useless = np.random.randint(2, size=[nb_points, nb_useless])
        xs = np.hstack((xs, useless))

    return Data(xs=xs, ys=ys, cs=mindsets)


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
