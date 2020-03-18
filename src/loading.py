from functools import partial

from src.config import DATASET_QUESTIONNAIRE_SYNTHETIC, DATASET_BINARY_IRIS, DATASET_SBM
from src.order_functions import implicit_order, cut_order
from src.datasets.loading.iris import get_binarized_iris
from src.datasets.loading.questionnaire import load_synthetic_questionnaire
from src.datasets.loading.sbm import load_sbm


def get_dataset(dataset):

    if dataset.type == DATASET_QUESTIONNAIRE_SYNTHETIC:
        xs, ys, cs = load_synthetic_questionnaire(dataset.path)
        order_function = partial(implicit_order, xs)
    if dataset.type == DATASET_BINARY_IRIS:
        xs, ys = get_binarized_iris()
        order_function = partial(implicit_order, xs)
    if dataset.type == DATASET_SBM:
        xs, ys = load_sbm(block_size=30, nb_blocks=3, p_in=.6, p_out=.1)
        order_function = partial(cut_order, xs)

    return xs, ys, order_function
