from functools import partial

from src.config import DATASET_QUESTIONNAIRE_SYNTHETIC, DATASET_BINARY_IRIS, DATASET_SBM, \
    DATASET_LFR, DATASET_RING_OF_CLIQUES
from src.datasets.graphs import load_RPG, load_LFR, load_ROC
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
    order_function: function
        The partially evaluated order function
    """

    if dataset.type == DATASET_QUESTIONNAIRE_SYNTHETIC:
        xs, ys, cs = load_synthetic_questionnaire(dataset.path)
        order_function = partial(implicit_order, xs)
    elif dataset.type == DATASET_BINARY_IRIS:
        xs, ys = get_binarized_iris()
        order_function = partial(implicit_order, xs)
    elif dataset.type == DATASET_SBM:
        xs, ys = load_RPG(block_size=10, nb_blocks=5, p_in=.9, p_out=.3)
        order_function = partial(cut_order, xs)
    elif dataset.type == DATASET_LFR:
        xs, ys = load_LFR(nb_nodes=50, tau1=3, tau2=1.5, mu=0.1,
                          min_community=10, average_degree=3, seed=10)
        order_function = partial(cut_order, xs)
    elif dataset.type == DATASET_RING_OF_CLIQUES:
        xs, ys = load_ROC(nb_cliques=20, clique_size=10)
        order_function = partial(cut_order, xs)

    return xs, ys, order_function
