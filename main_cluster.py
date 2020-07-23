import os
from functools import partial
from pathlib import Path
import cProfile
import re

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score, v_measure_score

import json
import pandas as pd

from src.order_functions import implicit_order
from src.parser import make_parser
from src.config import load_validate_settings, set_up_dirs
from src.execution import get_dataset_cuts_order
from src.plotting import get_position, plot_all_tangles
from src.tangle_tree import TangleTreeModel
from src.utils import get_points_to_plot, get_positions_from_labels

def exp(cost):
    return np.exp(-3*cost)

def sigmoid(cost):
    return 1 / (1 + np.exp(10 * (cost - 0.4)))

def main_tree(args):

    foundamental_parameters = {**args['experiment'], **args['dataset'], **args['preprocessing']}

    path = hash(json.dumps(foundamental_parameters, sort_keys=True))
    args["plot"]["path"] = "output/tree/" + str(path)

    if args['verbose'] >= 1:
        print(f'Working with parameters = {foundamental_parameters}', flush=True)

    data, orders, all_cuts = get_dataset_cuts_order(args)

    # run the tangle algorithm
    model = TangleTreeModel(agreement=args["experiment"]["agreement"], cuts=all_cuts["values"], costs=orders,
                            weight_fun=sigmoid, print_cuts=True)

    tangles = np.array(model.tangles)
    probs_tangles = np.stack(np.array(tangles[:, 0]))
    coordinate_tangles = tangles[:, 2]

    leaves = np.array(model.maximals)
    if len(leaves) > 0:
        probs_leaves = np.stack(np.array(leaves[:, 0]))
        coordinate = leaves[:, 1]
        tangle_labels = np.argmax(probs_leaves, axis=0)


        vms_tangles = v_measure_score(data["ys"], tangle_labels)
        homo_tangle = metrics.homogeneity_score(data["ys"], tangle_labels)
    else:
        homo_tangle = 0
        vms_tangles = 0

    print(vms_tangles, homo_tangle)

    if args["plot"]["tangles"]:
        if args["plot"]["path"]:
            print("new folder")
            os.mkdir(args["plot"]["path"])
        if args['plot']['only_leaves']:
            plot_all_tangles(data, probs_leaves, coordinate, save=args["plot"]["path"])
        else:
            plot_all_tangles(data, probs_tangles, coordinate_tangles, save=args["plot"]["path"])


if __name__ == '__main__':

    # Make parser, read inputs from command line and resolve paths

    parser = make_parser()
    args_parser = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    print(root_dir)
    args = load_validate_settings(args_parser, root_dir=root_dir)
    args = set_up_dirs(args, root_dir=root_dir)

    main_tree(args)
