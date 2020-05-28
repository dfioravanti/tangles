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
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import radius_neighbors_graph, kneighbors_graph

from src.order_functions import implicit_order, cut_order
from src.parser import make_parser
from src.config import load_validate_settings, set_up_dirs
from src.execution import get_dataset_cuts_order, tangle_computation, compute_cuts, order_cuts, pick_cuts_up_to_order
from src.plotting import get_position
from src.tangle_tree import TangleTreeModel
from src.utils import get_points_to_plot, get_positions_from_labels

def exp(cost):
    return np.exp(-3*cost)

def sigmoid(cost):
    return 1 / (1 + np.exp(10 * (cost - 0.4)))

def main_tree(args):

    data = {}

    foundamental_parameters = {**args['experiment'], **args['dataset'], **args['preprocessing']}

    unique_id = hash(json.dumps(foundamental_parameters, sort_keys=True))
    df_output = pd.DataFrame()

    if args['verbose'] >= 1:
        print(f'Working with parameters = {foundamental_parameters}', flush=True)

    #data, orders, all_cuts = get_dataset_cuts_order(args)

    xs, ys = make_moons(n_samples=args['dataset']['n_samples'],
                        noise=args['dataset']['noise'],
                        random_state=args['experiment']['seed'], shuffle=False)
    A = kneighbors_graph(xs, 30, mode="distance").toarray()

    sigma = np.median(A[A > 0])

    A[A > 0] = np.exp(- A[A > 0] / (2 * sigma ** 2))

    G = nx.from_numpy_matrix(A)

    data["xs"] = xs
    data["ys"] = ys
    data["A"] = A
    data["G"] = G

    order_function = partial(cut_order, A)

    cuts = compute_cuts(data, args, verbose=args['verbose'])

    cuts, orders = order_cuts(cuts, order_function)
    all_cuts, orders = pick_cuts_up_to_order(cuts, orders, percentile=args['experiment']['percentile_orders'])


    clustering = SpectralClustering(n_clusters=2, random_state=args["experiment"]["seed"]).fit(data["xs"])
    labels = clustering.labels_

    ars = adjusted_rand_score(data['ys'], labels)
    print("SC: ", ars)

    clustering = SpectralClustering(n_clusters=2, random_state=args["experiment"]["seed"]).fit(data["A"])
    labels = clustering.labels_

    ars = adjusted_rand_score(data['ys'], labels)
    print("SC A: ", ars)

    model = TangleTreeModel(agreement=args["experiment"]["agreement"], cuts=all_cuts["values"], costs=orders, weight_fun=sigmoid)

    tangles = np.array(model.tangles)

    probs = tangles[:, 0]
    cuts_original_tree = tangles[:, 1]
    coordinate = tangles[:, 2]

    leaves = np.array(model.maximals)

    print(leaves.shape)
    hard_clustering = np.argmax(leaves, axis=0)
    print(hard_clustering)
    ars = adjusted_rand_score(data['ys'], hard_clustering)
    print(ars)

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
