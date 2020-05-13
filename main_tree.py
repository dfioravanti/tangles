from functools import partial
from pathlib import Path

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import json
import pandas as pd

from src.order_functions import implicit_order
from src.parser import make_parser
from src.config import load_validate_settings, set_up_dirs
from src.execution import compute_clusters, compute_evaluation, get_dataset_cuts_order, tangle_computation, plotting, \
    compute_maximal_tangles, compute_clusters_maximals, print_tangles_names, tangles_to_range_answers, \
    compute_fuzzy_clusters, soft_plotting  # , compute_soft_evaluation
from src.plotting import get_position
from src.tangle_tree import TangleTree
from src.utils import get_points_to_plot


def main_tree(args):

    foundamental_parameters = {**args['experiment'], **args['dataset'], **args['preprocessing']}

    unique_id = hash(json.dumps(foundamental_parameters, sort_keys=True))
    df_output = pd.DataFrame()

    if args['verbose'] >= 1:
        print(f'Working with parameters = {foundamental_parameters}', flush=True)

    data, orders, all_cuts, name_cuts = get_dataset_cuts_order(args)
    max_order = orders.max()

    print("got all the data")

    # all_cuts = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    #             [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]]

    tree = TangleTree(agreement=args["experiment"]["agreement"], cuts=all_cuts)

    print(" --- Built the tree and now trying to find all the splitting tangles \n")
    print(" --- Get all the splitting tangles \n")
    tree.check_for_splitting_tangles()

    print(" -- printing original tree")
    tree.print()

    print(" -- printing condensed tree")
    tree.print(condensed=True)

    splitting = tree.nb_splitting_tangles
    leaves = tree.get_leaves(tree.root)
    print("# splitting tangles: ", splitting, "\n")
    print("# leaves: ", len(leaves), "\n")

    print("Calculating the p values for each node")
    tree.update_p()

    tangle_probs = tree.condensed_tree.tangles

    for l in tangle_probs:
        print("Orientation: ", l[1], "\n \t probabilities: ", l[0])

    plot = False
    if plot:
        plt.set_cmap("Blues")

        #pos, _ = get_points_to_plot(data["xs"], None)
        pos = get_position(data["G"], data["ys"])

        for l in tangle_probs:
            plt.figure()

            plt.title("layer: " + str(len(l[1])) + " - coordinate: " + str(np.array(l[2]).astype(int)))

            plt.scatter(pos[:, 0], pos[:, 1], c=l[0])
            plt.clim(0, 1)
            plt.colorbar()
            plt.savefig("output/tree/layer_" + str(len(l[1])) + ".png")


if __name__ == '__main__':

    # Make parser, read inputs from command line and resolve paths

    parser = make_parser()
    args_parser = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    print(root_dir)
    args = load_validate_settings(args_parser, root_dir=root_dir)
    args = set_up_dirs(args, root_dir=root_dir)

    main_tree(args)
