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
from src.execution import get_dataset_cuts_order, compute_cuts, order_cuts, pick_cuts_up_to_order
from src.plotting import get_position, plot_all_tangles
from src.tangle_tree import TangleTreeModel
from src.utils import get_points_to_plot, get_positions_from_labels

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

def main_tree(args):

    foundamental_parameters = {**args['experiment'], **args['dataset'], **args['preprocessing']}

    if args['verbose'] >= 1:
        print(f'Working with parameters = {foundamental_parameters}', flush=True)

    data, orders, all_cuts = get_dataset_cuts_order(args)
    data_names = ["Acinetobacter johnsonii",
                    "Acinetobacter lwoffii",
                    "Anaerococcus unclassified",
                    "Corynebacteriaceae unclassified",
                    "Corynebacterium kroppenstedtii",
                    "Corynebacterium tuberculostearicum",
                    "Corynebacterium tuberculostearicum C<0.8",
                    "Corynebacterium unclassified",
                    "Corynebacterium unclassified C<0.8",
                    "Dolosigranulum unclassified",
                    "Enterobacteriaceae unclassified C<0.8",
                    "Averyella C<0.8",
                    "Erwinia C<0.8",
                    "Leclercia C<0.8",
                    "Klebsiella C<0.8",
                    "Proteus unclassified",
                    "Proteus vulgaris",
                    "Proteus vulgaris C<0.8",
                    "Raoultella C<0.8",
                    "Serratia marcescens",
                    "unclassified Escherichia",
                    "Finegoldia unclassified",
                    "Moraxella unclassified",
                    "Moraxellaceae unclassified",
                    "Moraxellaceae unclassified C<0.8",
                    "Peptoniphilus asaccharolyticus",
                    "Propionibacterium acnes",
                    "Propionibacterium granulosum",
                    "Pseudomonas unclassified",
                    "Psychrobacter unclassified",
                    "Simonsiella unclassified",
                    "Simonsiella unclassified C<0.8",
                    "Staphylococcus aprophyticus C<0.8",
                    "Staphylococcus aureus",
                    "Staphylococcus aureus C<0.8",
                    "Staphylococcus auricularis C<0.8",
                    "Staphylococcus capitis C<0.8",
                    "Staphylococcus caprae C<0.8",
                    "Staphylococcus epidermidis",
                    "Staphylococcus epidermidis C<0.8",
                    "Staphylococcus haemolyticus C<0.8",
                    "Staphylococcus hominis C<0.8",
                    "Staphylococcus lugdunensis",
                    "Staphylococcus lugdunensis C<0.8",
                    "Staphylococcus pasteuri C<0.8",
                    "Staphylococcus pettenkoferi C<0.8",
                    "Staphylococcus warneri C<0.8",
                    "Stenotrophomonas unclassified",
                    "Streptococcus unclassified",
                    "Streptococcus unclassified C<0.8"]

    nb_clusters = len(np.unique(data["ys"]))

    fig, ax = plt.subplots(1, 1, figsize=(30, 7.5))
    for i, d in enumerate(data['xs'].T):
        ax.scatter(np.random.normal(i, 0.04, size=len(d)), d, alpha=0.1)
        ax.boxplot(d, positions=[i], showfliers=False, )

    plt.yscale('log')
    xtickNames = plt.setp(ax, xticklabels=data_names)
    plt.setp(xtickNames)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('output/microbiome_raw.png')

    return True


if __name__ == '__main__':

    # Make parser, read inputs from command line and resolve paths

    parser = make_parser()
    args_parser = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    print(root_dir)
    args = load_validate_settings(args_parser, root_dir=root_dir)
    args = set_up_dirs(args, root_dir=root_dir)

    main_tree(args)
