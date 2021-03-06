import numpy as np

from sklearn.neighbors import DistanceMetric


def implicit_cost(xs, n_samples, cut):
    """
    This function computes the implicit order of a cut.
    It is zero if the cut is either the whole set or the empty set

    If n_samples if not None we do a montecarlo approximation of the value.

    Parameters
    ----------
    xs : array of shape [n_points, n_features]
        The points in our space
    cut: array of shape [n_points]
        The cut that we are considering
    n_samples: int, optional (default=None)
        The maximums number of points to take per orientation in the Monte Carlo approximation of the order

    Returns
    -------
    expected_order, int
        The average order for the cut
    """

    _, n_features = xs.shape
    if np.all(cut) or np.all(~cut):
        return 0

    if n_samples == -1:

        in_cut = xs[cut, :]
        out_cut = xs[~cut, :]

    else:

        idx = np.arange(len(xs))

        if n_samples <= len(idx[cut]):
            idx_in = np.random.choice(idx[cut], size=n_samples, replace=False)
            in_cut = xs[idx_in, :]
        else:
            in_cut = xs[cut, :]

        if n_samples <= len(idx[~cut]):
            idx_out = np.random.choice(idx[~cut], size=n_samples, replace=False)
            out_cut = xs[idx_out, :]
        else:
            out_cut = xs[~cut, :]

    metric = DistanceMetric.get_metric('manhattan')

    distance = metric.pairwise(in_cut, out_cut)
    similarity = 1. / (distance / np.max(distance))
    expected_similarity = np.average(similarity)

    return np.round(expected_similarity, 2)


def edges_cut_cost(A, cut):

    """
    Compute the value of a graph cut, i.e. the number of vertex that are cut by the bipartition

    Parameters
    ----------
    A: array of shape [nb_vertices, nb_vertices]
        Adjacency matrix for our graph
    cut: array of shape [n_points]
        The cut that we are considering

    Returns
    -------
    order: int
        order of the cut
    """

    partition = np.where(cut == True)[0]
    comp = np.where(cut == False)[0]

    order = np.sum(A[np.ix_(partition, comp)])

    return order
