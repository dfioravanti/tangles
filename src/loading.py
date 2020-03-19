from functools import partial

from src.config import DATASET_QUESTIONNAIRE_SYNTHETIC, DATASET_BINARY_IRIS, DATASET_SBM, \
    DATASET_LFR, DATASET_RING_OF_CLIQUES
from src.datasets.graphs import load_RPG, load_LFR, load_ROC
from src.datasets.iris import get_binarized_iris
from src.datasets.questionnaire import load_synthetic_questionnaire
from src.order_functions import implicit_order, cut_order


def get_dataset(dataset):
    if dataset.type == DATASET_QUESTIONNAIRE_SYNTHETIC:
        xs, ys, cs = load_synthetic_questionnaire(dataset.path)
        order_function = partial(implicit_order, xs)
    elif dataset.type == DATASET_BINARY_IRIS:
        xs, ys = get_binarized_iris()
        order_function = partial(implicit_order, xs)
    elif dataset.type == DATASET_SBM:
        xs, ys = load_RPG(block_size=50, nb_blocks=2, p_in=.8, p_out=.2)
        order_function = partial(cut_order, xs)
    elif dataset.type == DATASET_LFR:
        xs, ys = load_LFR(nb_nodes=50, tau1=3, tau2=1.5, mu=0.1,
                          min_community=10, average_degree=3, seed=10)
        order_function = partial(cut_order, xs)
    elif dataset.type == DATASET_RING_OF_CLIQUES:
        xs, ys = load_ROC(nb_cliques=20, clique_size=10)
        order_function = partial(cut_order, xs)

    return xs, ys, order_function
