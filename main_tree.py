from functools import partial
from pathlib import Path

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp

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


def main_tree(args):

    foundamental_parameters = {**args['experiment'], **args['dataset'], **args['preprocessing']}

    unique_id = hash(json.dumps(foundamental_parameters, sort_keys=True))
    df_output = pd.DataFrame()

    if args['verbose'] >= 1:
        print(f'Working with parameters = {foundamental_parameters}', flush=True)

    data, orders, all_cuts, name_cuts = get_dataset_cuts_order(args)

    print("costs correspond to data?: ", np.shape(all_cuts), np.shape(orders))

    model = TangleTreeModel(agreement=args["experiment"]["agreement"], cuts=all_cuts, costs=orders)

    tangles = np.array(model.tangles)

    probs = tangles[:, 0]
    cuts_original_tree = tangles[:, 1]

    #for p, c in zip(probs, cuts_original_tree):
    #    print("Orientation: ", c, "\n \t probabilities: ", p)

    plot = True
    if plot:
        pos = get_positions_from_labels(data["ys"])
        for i, p in enumerate(probs):
            fig, (ax1, ax2) = plt.subplots(1, 2)

            _ = ax1.scatter(pos[:, 0],
                            pos[:, 1],
                            c=data["ys"],
                            cmap="tab20")

            col2 = ax2.scatter(pos[:, 0],
                               pos[:, 1],
                               c=p,
                               cmap="Blues")

            col2.set_clim(0, 1)

            plt.colorbar(col2, ax=ax2)
            plt.savefig("output/tree/plot_" + str(i) + ".png")

if __name__ == '__main__':

    # Make parser, read inputs from command line and resolve paths

    parser = make_parser()
    args_parser = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    print(root_dir)
    args = load_validate_settings(args_parser, root_dir=root_dir)
    args = set_up_dirs(args, root_dir=root_dir)

    main_tree(args)
