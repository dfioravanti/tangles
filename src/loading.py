from functools import partial
import numpy as np

from src.config import DATASET_QUESTIONNAIRE_SYNTHETIC, DATASET_BINARY_IRIS, DATASET_SBM, \
    DATASET_MULTILEVEL, DATASET_RING_OF_CLIQUES, DATASET_FLORENCE, DATASET_BIG5, DATASET_KNN
from src.datasets.big5 import load_BIG5
from src.datasets.graphs import load_RPG, load_LFR, load_ROC, load_FLORENCE, load_multilevel
from src.datasets.iris import get_binarized_iris
from src.datasets.kNN import load_KNN
from src.datasets.questionnaire import make_synthetic_questionnaire
from src.order_functions import implicit_order, cut_order


def get_dataset_and_order_function(args, seed):

    """
    TODO: MOVE A LOT OF THESE PARAMETERS OUT AND GET THEM VIA COMMAND LINE

    Function that returns the desired dataset and the order function in the format that we expect.
    Datasets are always in the format of
        - xs: Features that we need for clustering, like questions for the questionnaire or the adjacency matrix for
              the graph
        - ys: Class label
    Order functions are assumed to be functions that only need a bipartition as inputs and return the order
    of that bipartion. We assume that all the other parameters are loaded via partial evaluation in this function.

    Parameters
    ----------
    dataset: SimpleNamespace
        The parameters of the dataset to load
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
    G = None
    if args.experiment.dataset_name == DATASET_QUESTIONNAIRE_SYNTHETIC:
        xs, ys, cs = make_synthetic_questionnaire(n_samples=args.q_syn.n_samples,
                                                  n_features=args.q_syn.n_features,
                                                  n_mindsets=args.q_syn.n_mindsets,
                                                  tolerance=args.q_syn.tolerance,
                                                  seed=seed,
                                                  centers=True)
        order_function = partial(implicit_order, xs, None)
    elif args.experiment.dataset_name == DATASET_BINARY_IRIS:
        xs, ys = get_binarized_iris()
        order_function = partial(implicit_order, xs, None)
    elif args.experiment.dataset_name == DATASET_BIG5:
        xs, ys = load_BIG5(args.big5.path)
        order_function = partial(implicit_order, xs, 100)
    elif args.experiment.dataset_name == DATASET_SBM:
        xs, ys, G = load_RPG(block_size=args.sbm.block_size, nb_blocks=args.sbm.nb_blocks,
                             p_in=args.sbm.p, p_out=args.sbm.q)
        order_function = partial(cut_order, xs)
    elif args.experiment.dataset_name == DATASET_MULTILEVEL:
        xs, ys, G = load_multilevel(nb_nodes=args.multilevel.block_size,
                                    p_in=args.multilevel.p_in,
                                    p_out=args.multilevel.p_out)
        order_function = partial(cut_order, xs)
    elif args.experiment.dataset_name == DATASET_RING_OF_CLIQUES:
        xs, ys, G = load_ROC(nb_cliques=args.roc.nb_cliques, clique_size=args.roc.clique_size)
        order_function = partial(cut_order, xs)
    elif args.experiment.dataset_name == DATASET_FLORENCE:
        xs, ys, G = load_FLORENCE()
        order_function = partial(cut_order, xs)
    elif args.experiment.dataset_name == DATASET_KNN:
        xs, ys, G = load_KNN(args.knn.mus, args.knn.vars, args.knn.block_size, args.knn.nb_blocks, args.knn.k)
        order_function = partial(cut_order, xs)

    return xs, ys, G, order_function
