from functools import partial

from src.config import DATASET_QUESTIONNAIRE_SYNTHETIC, DATASET_SBM, \
    DATASET_RING_OF_CLIQUES, DATASET_FLORENCE, DATASET_BIG5, DATASET_KNN_BLOBS
from src.datasets.big5 import load_BIG5
from src.datasets.graphs import load_RPG, load_ROC, load_FLORENCE
from src.datasets.kNN import load_knn_blobs
from src.datasets.questionnaire import make_synthetic_questionnaire
from src.order_functions import implicit_order, cut_order


def get_dataset_and_order_function(dataset_name, parameters):
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

    data = {}

    if dataset_name== DATASET_QUESTIONNAIRE_SYNTHETIC:
        xs, ys, cs = make_synthetic_questionnaire(n_samples=args.q_syn.n_samples,
                                                  n_features=args.q_syn.n_features,
                                                  n_mindsets=args.q_syn.n_mindsets,
                                                  tolerance=args.q_syn.tolerance,
                                                  seed=seed,
                                                  centers=True)

        data['xs'] = xs
        data['ys'] = ys
        order_function = partial(implicit_order, xs, None)
    elif dataset_name== DATASET_BIG5:
        xs, ys = load_BIG5(args.big5.path)

        data['xs'] = xs
        data['ys'] = ys
        order_function = partial(implicit_order, xs, 100)
    elif dataset_name == DATASET_SBM:
        A, ys, G = load_RPG(block_sizes=parameters['block_sizes'],
                            p_in=parameters['p'],
                            p_out=parameters['q'],
                            seed=parameters['seed'])

        data['A'] = A
        data['ys'] = ys
        data['G'] = G
        order_function = partial(cut_order, A)
    elif dataset_name== DATASET_RING_OF_CLIQUES:
        A, ys, G = load_ROC(nb_cliques=args.roc.nb_cliques, clique_size=args.roc.clique_size)

        data['A'] = A
        data['ys'] = ys
        data['G'] = G
        order_function = partial(cut_order, A)
    elif dataset_name== DATASET_FLORENCE:
        A, ys, G = load_FLORENCE()

        data['A'] = A
        data['ys'] = ys
        data['G'] = G
        order_function = partial(cut_order, A)
    elif dataset_name== DATASET_KNN_BLOBS:
        xs, ys, A, G = load_knn_blobs(blob_sizes=parameters['blob_sizes'],
                                      blob_centers=parameters['blobs_center'],
                                      k=parameters['k'],
                                      seed=parameters['seed'])

        data['xs'] = xs
        data['ys'] = ys
        data['A'] = A
        data['G'] = G
        order_function = partial(cut_order, A)

    return data, order_function
