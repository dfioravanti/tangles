from functools import partial

from src.config import DATASET_QUESTIONNAIRE_SYNTHETIC, DATASET_BINARY_IRIS, DATASET_SBM, \
    DATASET_LFR, DATASET_RING_OF_CLIQUES, DATASET_FLORENCE, DATASET_BIG5
from src.datasets.big5 import load_BIG5
from src.datasets.graphs import load_RPG, load_LFR, load_ROC, load_FLORENCE
from src.datasets.iris import get_binarized_iris
from src.datasets.questionnaire import load_synthetic_questionnaire
from src.order_functions import implicit_order, cut_order


def get_dataset_and_order_function(dataset):

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
    if dataset.name == DATASET_QUESTIONNAIRE_SYNTHETIC:
        xs, ys, cs = load_synthetic_questionnaire(dataset.path)
        order_function = partial(implicit_order, xs, None)
    elif dataset.name == DATASET_BINARY_IRIS:
        xs, ys = get_binarized_iris()
        order_function = partial(implicit_order, xs, None)
    elif dataset.name == DATASET_SBM:
        xs, ys, G = load_RPG(block_size=dataset.block_size, nb_blocks=dataset.nb_blocks,
                             p_in=dataset.p, p_out=dataset.q)
        order_function = partial(cut_order, xs)
    elif dataset.name == DATASET_LFR:
        xs, ys, G = load_LFR(nb_nodes=50, tau1=3, tau2=1.5, mu=0.1,
                          min_community=10, average_degree=3, seed=10)
        order_function = partial(cut_order, xs)
    elif dataset.name == DATASET_RING_OF_CLIQUES:
        xs, ys, G = load_ROC(nb_cliques=dataset.nb_cliques, clique_size=dataset.clique_size)
        order_function = partial(cut_order, xs)
    elif dataset.name == DATASET_FLORENCE:
        xs, ys, G = load_FLORENCE()
        order_function = partial(cut_order, xs)
    elif dataset.name == DATASET_BIG5:
        xs, ys = load_BIG5(dataset.path)
        order_function = partial(implicit_order, xs, 100)

    return xs, ys, G, order_function
