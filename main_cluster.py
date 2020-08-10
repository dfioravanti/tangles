import os
from copy import deepcopy
from functools import partial
from pathlib import Path
import cProfile
import re

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
from sklearn import metrics
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, v_measure_score

import json
import pandas as pd

from loading import get_dataset_and_order_function
from src.order_functions import implicit_order
from src.parser import make_parser
from src.config import load_validate_settings, set_up_dirs
from src.execution import get_dataset_cuts_order, compute_cuts
from src.plotting import get_position, plot_all_tangles
from src.tangle_tree import TangleTreeModel
from src.utils import get_points_to_plot, get_positions_from_labels

def exp(cost):
    return np.exp(-3*cost)

def sigmoid(cost):
    return 1 / (1 + np.exp(10 * (cost - 0.4)))

def main_tree(args):

    compare = True

    foundamental_parameters = {**args['experiment'], **args['dataset'], **args['preprocessing']}

    path = hash(json.dumps(foundamental_parameters, sort_keys=True))
    args["plot"]["path"] = "output/tree/" + "microbiome_use_features"

    if args['verbose'] >= 1:
        print(f'Working with parameters = {foundamental_parameters}', flush=True)

    data, orders, all_cuts = get_dataset_cuts_order(args)

    #data, order_function = get_dataset_and_order_function(args)
    #data, orders, all_cuts = get_dataset_cuts_order(args)

    #orders =np.array([0.1, 0.11, 1.0, 1.3, 2.0, 2.1])

    # for i, c in enumerate(all_cuts['values']):
    #     fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    #     plt.axis('off')
    #     plt.grid(False)
    #     _ = ax.scatter(data['xs'][:, 0],
    #                    data['xs'][:, 1],
    #                    c=c,
    #                    cmap="Blues",
    #                    vmin=-0.5,
    #                    vmax=1,
    #                    s=10)
    #     plt.savefig(str(args["plot"]["path"]) + "/cut_{}_t.pdf".format(i))
    # 
    #     fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    #     plt.axis('off')
    #     plt.grid(False)
    #     _ = ax.scatter(data['xs'][:, 0],
    #                    data['xs'][:, 1],
    #                    c=~c,
    #                    cmap="Blues",
    #                    vmin=-0.5,
    #                    vmax=1,
    #                    s=10)
    #     plt.savefig(str(args["plot"]["path"]) + "/cut_{}_f.pdf".format(i))

    # run the tangle algorithm
    model = TangleTreeModel(agreement=args["experiment"]["agreement"], cuts=deepcopy(all_cuts["values"]), costs=orders,
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
        #vms_tangles = adjusted_rand_score(data["ys"], tangle_labels)
        homo_tangle = metrics.homogeneity_score(data["ys"], tangle_labels)
    else:
        homo_tangle = 0
        vms_tangles = 0

    print(vms_tangles, homo_tangle)
    pos, _ = get_points_to_plot(data["xs"])

    if args["plot"]["tangles"]:
        if args["plot"]["path"]:
            print("new folder")
            if not os.path.exists(args["plot"]["path"]):
                os.mkdir(args["plot"]["path"])

            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            plt.axis('off')
            plt.grid(False)
            fig.patch.set_facecolor('grey')
            _ = ax.scatter(pos[:, 0],
                           pos[:, 1],
                           c=tangle_labels,
                           cmap="Set2",
                           s=10)
            ax.set_title('v score : {}'.format(np.round(vms_tangles, 2)))
            plt.savefig(str(args["plot"]["path"]) + "/plot_clustering.pdf")


            if compare:
                nb_clusters = len(np.unique(data['ys']))
                spectral_label = SpectralClustering(n_clusters=nb_clusters, affinity='nearest_neighbors', n_neighbors=100, assign_labels="kmeans",random_state=0).fit(data['xs']).labels_
                spectral_vms = v_measure_score(data["ys"], spectral_label)
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                plt.axis('off')
                plt.grid(False)
                fig.patch.set_facecolor('grey')
                _ = ax.scatter(pos[:, 0],
                               pos[:, 1],
                               c=spectral_label,
                               cmap="Set2",
                               s=10)

                ax.set_title('v score : {}'.format(np.round(spectral_vms, 2)))
                plt.savefig(str(args["plot"]["path"]) + "/plot_spectral.pdf")

                kMeans_label = KMeans(n_clusters=nb_clusters, random_state=0).fit(data['xs']).labels_
                kMeans_vms = v_measure_score(data["ys"], kMeans_label)
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                plt.axis('off')
                plt.grid(False)
                fig.patch.set_facecolor('grey')
                _ = ax.scatter(pos[:, 0],
                               pos[:, 1],
                               c=kMeans_label,
                               cmap="Set2",
                               s=10)

                ax.set_title('v score : {}'.format(np.round(kMeans_vms, 2)))
                plt.savefig(str(args["plot"]["path"]) + "/plot_k_means.pdf")

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
