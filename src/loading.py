from functools import partial


from src.config import DATASET_BINARY_QUESTIONNAIRE, DATASET_SBM, DATASET_QUESTIONNAIRE, \
    DATASET_POLITICAL_BOOKS, DATASET_FLORENCE, DATASET_BIG5, DATASET_KNN_BLOBS, DATASET_CANCER, DATASET_CANCER10
from src.datasets.big5 import load_BIG5
from src.datasets.cancer import load_CANCER
from src.datasets.cancer10 import load_CANCER10
from src.datasets.graphs import load_RPG, load_POLI_BOOKS, load_FLORENCE
from src.datasets.kNN import load_knn_blobs
from src.datasets.questionnaire import make_binary_questionnaire, make_questionnaire
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

    if args['experiment']['dataset_name'] == DATASET_BINARY_QUESTIONNAIRE:
        xs, ys, cs = make_binary_questionnaire(n_samples=args['dataset']['n_samples'],
                                               n_features=args['dataset']['n_features'],
                                               n_mindsets=args['dataset']['n_mindsets'],
                                               n_mistakes=args['dataset']['n_mistakes'],
                                               seed=args['experiment']['seed'],
                                               centers=True)

        data['xs'] = xs
        data['ys'] = ys
        order_function = partial(implicit_order, xs, None)
    elif args['experiment']['dataset_name'] == DATASET_QUESTIONNAIRE:
        xs, ys = make_questionnaire(n_samples=args['dataset']['n_samples'],
                                               n_features=args['dataset']['n_features'],
                                               n_mindsets=args['dataset']['n_mindsets'],
                                               range_answers=args['dataset']['range_answers'],
                                               seed=args['experiment']['seed'])

        data['xs'] = xs
        data['ys'] = ys
        order_function = partial(implicit_order, xs, None)
    elif args['experiment']['dataset_name'] == DATASET_CANCER:
        xs, ys = load_CANCER(args['dataset']['nb_bins'])

        data['xs'] = xs
        data['ys'] = ys
        order_function = partial(implicit_order, xs, None)
    elif args['experiment']['dataset_name'] == DATASET_BIG5:
        xs, ys = load_BIG5(args['dataset']['path'])

        data['xs'] = xs
        data['ys'] = ys
        order_function = partial(implicit_order, xs, None)
    elif args['experiment']['dataset_name'] == DATASET_CANCER10:
        xs, ys = load_CANCER10(args['dataset']['path'])

        data['xs'] = xs
        data['ys'] = ys
        order_function = partial(implicit_order, xs, None)
    elif args['experiment']['dataset_name'] == DATASET_SBM:
        A, ys, G = load_RPG(block_sizes=args['dataset']['block_sizes'],
                            p_in=args['dataset']['p'],
                            p_out=args['dataset']['q'],
                            seed=args['experiment']['seed'])

        data['A'] = A
        data['ys'] = ys
        data['G'] = G
        order_function = partial(cut_order, A)
    elif args['experiment']['dataset_name'] == DATASET_POLITICAL_BOOKS:
        A, ys, G = load_POLI_BOOKS(path_nodes=args['dataset']['path_nodes'],
                                   path_edges=args['dataset']['path_edges'])

        data['A'] = A
        data['ys'] = ys
        data['G'] = G
        order_function = partial(cut_order, A)
    elif args['experiment']['dataset_name'] == DATASET_FLORENCE:
        A, ys, G = load_FLORENCE()

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
