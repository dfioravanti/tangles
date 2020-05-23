from functools import partial
from pathlib import Path
import cProfile
import re

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
from sklearn.metrics import adjusted_rand_score

import json
import pandas as pd

from src.order_functions import implicit_order
from src.parser import make_parser
from src.config import load_validate_settings, set_up_dirs
from src.execution import compute_clusters, compute_evaluation, get_dataset_cuts_order, tangle_computation, plotting, \
    compute_maximal_tangles, compute_clusters_maximals, print_tangles_names, tangles_to_range_answers, \
    compute_fuzzy_clusters, soft_plotting  # , compute_soft_evaluation
from src.plotting import get_position
from src.tangle_tree import TangleTreeModel
from src.utils import get_points_to_plot, get_positions_from_labels

def exp(cost):
    return np.exp(-6*cost)

def sigmoid(cost):
    return 1 / (1 + np.exp(10 * (cost - 0.4)))

def main_tree(args):

    foundamental_parameters = {**args['experiment'], **args['dataset'], **args['preprocessing']}

    unique_id = hash(json.dumps(foundamental_parameters, sort_keys=True))
    df_output = pd.DataFrame()

    if args['verbose'] >= 1:
        print(f'Working with parameters = {foundamental_parameters}', flush=True)

    data, orders, all_cuts, name_cuts = get_dataset_cuts_order(args)

    model = TangleTreeModel(agreement=args["experiment"]["agreement"], cuts=all_cuts, costs=orders, weight_fun=sigmoid)

    tangles = np.array(model.tangles)

    probs = np.stack(tangles[:, 0])[1:]
    cuts_original_tree = tangles[:, 1]
    coordinate = tangles[:, 2]

    hard_clustering = np.argmax(probs, axis=0)
    print(hard_clustering)
    ars = adjusted_rand_score(data['ys'], hard_clustering)
    print(ars)

    result = pd.Series({**foundamental_parameters,  'ARS': ars}).to_frame().T
    path = args['output_dir'] / f'evaluation_{unique_id}.csv'
    result.to_csv(path)

if __name__ == '__main__':

    # Make parser, read inputs from command line and resolve paths

    parser = make_parser()
    args_parser = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    print(root_dir)
    args = load_validate_settings(args_parser, root_dir=root_dir)
    args = set_up_dirs(args, root_dir=root_dir)

    main_tree(args)
