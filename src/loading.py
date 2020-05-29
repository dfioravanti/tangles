from functools import partial

import pandas as pd

from sklearn.neighbors import radius_neighbors_graph
from sklearn.datasets import make_moons

import networkx as nx

from src.config import DATASET_SBM, DATASET_LFR, DATASET_QUESTIONNAIRE, DATASET_BLOBS, DATASET_CANCER, DATASET_CANCER10, \
    DATASET_MINDSETS, DATASET_RETINAL, DATASET_MIES, DATASET_MOONS, DATASET_WAWE, DATASET_KNN_GAUSS_BLOBS

from src.datasets.cancer import load_CANCER
from src.datasets.cancer10 import load_CANCER10
from src.datasets.graphs import load_SBM, load_LFR
from src.datasets.kNN import load_blobs
from src.datasets.mies import load_MIES
from src.datasets.mindsets import make_mindsets
from src.datasets.questionnaire import make_questionnaire
from src.datasets.retinal import load_RETINAL
from src.datasets.gauss_blob import load_knn_gauss_blobs
from src.order_functions import implicit_order, cut_order


def get_dataset_and_order_function(args):
    """

    Function that returns the desired dataset and the order function in the format that we expect.
    Datasets are always in the format of
        - xs: Features that we need for clustering, like questions for the questionnaire or the adjacency matrix for
              the graph
        - ys: Class label
    Order functions are assumed to be functions that only need a bipartition as inputs and return the order
    of that bipartion. We assume that all the other args['dataset'] are loaded via partial evaluation in this function.

    args['dataset']
    ----------
    dataset: SimpleNamespace
        The args['dataset'] of the dataset to load
    seed: int
        The seed for the RNG

    Returns
    -------
    xs: array of shape [n_points, n_features] or array of shape [n_points, n_points]
        The points in our space or an adjacency matrix
    ys: array of shape [n_points]
        The array of class labels
    G: Graph or None
        The graph associated with the adjacency matrix
    order_function: function
        The partially evaluated order function
    """

    data = {}
    data['xs'] = None
    data['ys'] = None
    data['cs'] = None
    data['A'] = None
    data['G'] = None

    if args['experiment']['dataset_name'] == DATASET_MINDSETS:
        xs, ys, cs = make_mindsets(mindset_sizes=args['dataset']['mindset_sizes'],
                                   nb_questions=args['dataset']['nb_questions'],
                                   nb_useless=args['dataset']['nb_useless'],
                                   noise=args['dataset']['noise'],
                                   seed=args['experiment']['seed'])
        data['xs'] = xs
        data['ys'] = ys
        data['cs'] = cs
        order_function = partial(implicit_order, xs, None)
    elif args['experiment']['dataset_name'] == DATASET_QUESTIONNAIRE:
        xs, ys, cs = make_questionnaire(nb_samples=args['dataset']['nb_samples'],
                                        nb_features=args['dataset']['nb_features'],
                                        nb_mindsets=args['dataset']['nb_mindsets'],
                                        centers=args['dataset']['centers'],
                                        range_answers=args['dataset']['range_answers'],
                                        seed=args['experiment']['seed'])

        data['xs'] = xs
        data['ys'] = ys
        data['cs'] = cs
        order_function = partial(implicit_order, xs, None)
    elif args['experiment']['dataset_name'] == DATASET_RETINAL:
        xs, ys = load_RETINAL(root_path=args['root_dir'],
                              nb_bins=args['dataset']['nb_bins'],
                              max_idx=args['dataset']['max_idx'])

        data['xs'] = xs
        data['ys'] = ys
        order_function = partial(implicit_order, xs, 200)
    elif args['experiment']['dataset_name'] == DATASET_MOONS:
        xs, ys = make_moons(n_samples=args['dataset']['n_samples'],
                            noise=args['dataset']['noise'],
                            random_state=args['experiment']['seed'], shuffle=False)
        A = radius_neighbors_graph(xs, radius=args['dataset']['radius'], mode="connectivity").toarray()
        G = nx.from_numpy_matrix(A)
        data['xs'] = xs
        data['ys'] = ys
        data['A'] = A
        data['G'] = G
        order_function = partial(cut_order, A)
    elif args['experiment']['dataset_name'] == DATASET_CANCER:
        xs, ys = load_CANCER(args['dataset']['nb_bins'])

        data['xs'] = xs
        data['ys'] = ys
        order_function = partial(implicit_order, xs, None)
    elif args['experiment']['dataset_name'] == DATASET_SBM:
        A, ys, G = load_SBM(block_sizes=args['dataset']['block_sizes'],
                            p_in=args['dataset']['p'],
                            p_out=args['dataset']['q'],
                            seed=args['experiment']['seed'])

        data['A'] = A
        data['ys'] = ys
        data['G'] = G
        order_function = partial(cut_order, A)
    elif args['experiment']['dataset_name'] == DATASET_BLOBS:
        xs, ys, A, G = load_blobs(blob_sizes=args['dataset']['blob_sizes'],
                                      blob_centers=args['dataset']['blob_centers'],
                                      sigma=args['dataset']['blob_variances'],
                                      k=args['dataset']['k'],
                                      seed=args['experiment']['seed'])

        data['xs'] = xs
        data['ys'] = ys
        data['A'] = A
        data['G'] = G
        order_function = partial(implicit_order, xs, None)
        order_function = partial(cut_order, A)

        
    elif args['experiment']['dataset_name'] == DATASET_LFR:
        A, ys, G = load_LFR(nb_nodes=args['dataset']['nb_nodes'],
                            tau1=args['dataset']['tau1'],
                            tau2=args['dataset']['tau2'],
                            mu=args['dataset']['mu'],
                            average_degree=args['dataset']['average_degree'],
                            min_community=args['dataset']['min_community'],
                            seed=args['experiment']['seed'])

        data['A'] = A
        data['ys'] = ys
        data['G'] = G
        order_function = partial(cut_order, A)
        
    elif args['experiment']['dataset_name'] == DATASET_WAWE:
        df = pd.read_csv('datasets/waveform.csv')
        xs = df[df.columns[:-1]].to_numpy()
        ys = df[df.columns[-1]].to_numpy()

        data['xs'] = xs
        data['ys'] = ys
        
        order_function = partial(implicit_order, xs, 300)

    elif args['experiment']['dataset_name'] == DATASET_KNN_GAUSS_BLOBS:
        xs, ys, A, G = load_knn_gauss_blobs(blob_sizes=args['dataset']['blob_sizes'],
                                      blob_centers=args['dataset']['blob_centers'],
                                      blob_variances=args['dataset']['sigma'],
                                      k=args['dataset']['k'],
                                      seed=args['experiment']['seed'])

        data['xs'] = xs
        data['A'] = A
        data['ys'] = ys
        data['G'] = G

    return data, order_function

    