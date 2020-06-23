from pathlib import Path
import os

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
import pandas as pd

import json
import pandas as pd
import sklearn
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import radius_neighbors_graph, kneighbors_graph

from src.loading import get_dataset_and_order_function
from src.order_functions import implicit_order, cut_order
from src.parser import make_parser
from src.config import load_validate_settings, set_up_dirs
from src.execution import get_dataset_cuts_order, tangle_computation, compute_cuts, order_cuts, pick_cuts_up_to_order
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

    runs = 10

    current_df = pd.DataFrame(columns=["dataset_name", "agreement", "percentile", "spectral_ars", "spectral_std",
                               "kmeans_ars", "kmeans_std",
                               "tangle_ars", "tangle_std"])

    parameter_df = pd.DataFrame([args["dataset"]])

    current_df = pd.concat([current_df, parameter_df], ignore_index=True)

    current_df.dataset_name = args["experiment"]["dataset_name"]
    current_df.agreement = args["experiment"]["agreement"]
    current_df.percentile = args["experiment"]["percentile_orders"]

    ars_spectral = np.zeros(runs)
    ars_kmeans = np.zeros(runs)
    ars_tangles = np.zeros(runs)
    for r in range(runs):
        args["experiment"]["seed"] = np.random.randint(100)

        ars_spectral_tmp, ars_kmeans_tmp, ars_tangles_tmp = run(args)

        ars_spectral[r] = ars_spectral_tmp
        ars_kmeans[r] = ars_kmeans_tmp
        ars_tangles[r] = ars_tangles_tmp

    current_df.spectral_ars = ars_spectral.mean()
    current_df.kmeans_ars = ars_kmeans.mean()
    current_df.tangle_ars = ars_tangles.mean()

    current_df.spectral_std = ars_spectral.std()
    current_df.kmeans_std = ars_kmeans.std()
    current_df.tangle_std = ars_tangles.std()

    old_df = pd.read_csv("results.csv", sep=";")

    new_df = old_df.append(current_df, ignore_index=True)

    new_df.to_csv("results.csv", sep=";", index=False)

    return True


def run(args):

    # load the data
    data, orders, all_cuts = get_dataset_cuts_order(args)

    nb_clusters = len(np.unique(data["ys"]))

    # run the tangle algorithm o
    model = TangleTreeModel(agreement=args["experiment"]["agreement"], cuts=all_cuts["values"], costs=orders,
                            weight_fun=sigmoid)


    # evaluate tangle output
    leaves = np.array(model.maximals)
    if len(leaves) > 0:
        probs_leaves = np.stack(np.array(leaves[:, 0]))
        coordinate = leaves[:, 1]
        tangle_labels = np.argmax(probs_leaves, axis=0)

        ars_tangles = adjusted_rand_score(data["ys"], tangle_labels)
    else:

        ars_tangles = 0

    kmeans_labels = sklearn.cluster.KMeans(n_clusters=nb_clusters).fit(data["xs"]).labels_

    ars_kmeans = adjusted_rand_score(data["ys"], kmeans_labels)

    spectral_labels = sklearn.cluster.SpectralClustering(n_clusters=nb_clusters, affinity="nearest_neighbors").fit(data["xs"]).labels_
    #spectral_labels = sklearn.cluster.SpectralClustering(n_clusters=nb_clusters).fit(data["xs"]).labels_

    ars_spectral = adjusted_rand_score(data["ys"], spectral_labels)

    print(ars_tangles)

    if args["plot"]["tangles"]:
        if args["plot"]["path"]:
            print("new folder")
            os.mkdir(args["plot"]["path"])
        plot_all_tangles(data, probs_leaves, coordinate, save=args["plot"]["path"])

    return ars_spectral, ars_kmeans, ars_tangles

if __name__ == '__main__':

    # Make parser, read inputs from command line and resolve paths

    parser = make_parser()
    args_parser = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    print(root_dir)
    args = load_validate_settings(args_parser, root_dir=root_dir)
    args = set_up_dirs(args, root_dir=root_dir)

    main_tree(args)
