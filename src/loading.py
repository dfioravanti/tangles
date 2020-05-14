from functools import partial

from src.config import DATASET_SBM, DATASET_QUESTIONNAIRE, DATASET_KNN_BLOBS, DATASET_CANCER, DATASET_CANCER10, \
    DATASET_MINDSETS, DATASET_RETINAL, DATASET_MIES
from src.datasets.cancer import load_CANCER
from src.datasets.cancer10 import load_CANCER10
from src.datasets.graphs import load_SBM
from src.datasets.kNN import load_knn_blobs
from src.datasets.mies import load_MIES
from src.datasets.mindsets import make_mindsets
from src.datasets.questionnaire import make_questionnaire
from src.datasets.retinal import load_RETINAL
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
    elif args['experiment']['dataset_name'] == DATASET_MIES:
        xs, ys = load_MIES(root_path=args['root_dir'])

        data['xs'] = xs
        data['ys'] = ys
        order_function = partial(implicit_order, xs, False)
    elif args['experiment']['dataset_name'] == DATASET_CANCER:
        xs, ys = load_CANCER(args['dataset']['nb_bins'])

        data['xs'] = xs
        data['ys'] = ys
        order_function = partial(implicit_order, xs, None)
    elif args['experiment']['dataset_name'] == DATASET_CANCER10:
        xs, ys = load_CANCER10(args['dataset']['path'])

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
    elif args['experiment']['dataset_name'] == DATASET_KNN_BLOBS:
        xs, ys, A, G = load_knn_blobs(blob_sizes=args['dataset']['blob_sizes'],
                                      blob_centers=args['dataset']['blob_centers'],
                                      k=args['dataset']['k'],
                                      seed=args['experiment']['seed'])

        data['xs'] = xs
        data['ys'] = ys
        data['A'] = A
        data['G'] = G
        order_function = partial(cut_order, A)

    return data, order_function
