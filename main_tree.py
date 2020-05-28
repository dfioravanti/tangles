from functools import partial
from pathlib import Path
import cProfile
import re

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp

import json
import pandas as pd

from src.order_functions import implicit_order
from src.parser import make_parser
from src.config import load_validate_settings, set_up_dirs
from src.execution import get_dataset_cuts_order, tangle_computation
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

    percentile = args["experiment"]["percentile_orders"]
    idx = orders < np.max(orders) * percentile / 100.0

    orders = orders[idx]
    all_cuts = all_cuts[idx]

    print("Using up to ", percentile/100, " of the orders yielding ", len(orders), " cuts")

    model = TangleTreeModel(agreement=args["experiment"]["agreement"], cuts=all_cuts, costs=orders, weight_fun=sigmoid)

    tangles = np.array(model.tangles)

    probs = tangles[:, 0]
    cuts_original_tree = tangles[:, 1]
    coordinate = tangles[:, 2]

    #for p, c in zip(probs, cuts_original_tree):
    #    print("Orientation: ", c, "\n \t probabilities: ", p)

    plot = True
    if plot:
        #pos = get_positions_from_labels(data["ys"])
        #pos,_ = get_points_to_plot(data["xs"])

        #pos = nx.spring_layout(data["G"])
        #pos = np.array(list(pos.values()))

        pos = data["xs"]

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        plt.axis('off')
        plt.grid(False)
        fig.patch.set_facecolor('grey')
        _ = ax.scatter(pos[:, 0],
                       pos[:, 1],
                       c=data["ys"],
                       cmap="Set2")

        plt.savefig("output/tree/plot_gt.svg")

        for i, p in enumerate(probs):
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            plt.axis("off")
            plt.grid(False)
            fig.patch.set_facecolor('grey')

            # plt.suptitle("PARAMS "
            #              "knn_blobs: "
            #              + str(args["dataset"]["blob_sizes"])
            #              + "  a: "
            #              + str(args["experiment"]["agreement"])
            #              + "  pos: " + str(coordinate[i])
            #              + "  cost: " + str(orders[len(cuts_original_tree[i])]))

            col = ax.scatter(pos[:, 0],
                               pos[:, 1],
                               c=p,
                               cmap="Blues")

            col.set_clim(0, 1)

            plt.colorbar(col, ax=ax)
            plt.savefig("output/tree/plot_" + str(coordinate[i]) + ".svg")

if __name__ == '__main__':

    # Make parser, read inputs from command line and resolve paths

    parser = make_parser()
    args_parser = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    print(root_dir)
    args = load_validate_settings(args_parser, root_dir=root_dir)
    args = set_up_dirs(args, root_dir=root_dir)

    main_tree(args)
