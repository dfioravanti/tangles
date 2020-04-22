import numpy as np
import pandas as pd

from scipy.linalg import hadamard

from sklearn.datasets import make_blobs


def make_centers(n_features=20, n_mindsets=2):

    centers = np.zeros((n_mindsets, n_features), dtype=bool)
    idxs = np.array_split(np.arange(n_features), n_mindsets)
    idxs = idxs

    for i, idx in enumerate(idxs):

        centers[i, idx] = True

    return centers


def make_centers2(n_features=20, n_mindsets=2):

    """
    Generates n_mindsets blob_centers that are have hamming distance bigger than
    n_features // 2 - 2. This statement should be correct but I did not prove it.

    For now we generate those point by cropping a hadamarn matrix.
    TODO: Check out if we can use some other type of code that have a guaranteed hamming distance.
    TODO: It does not really scale up too much so better fix that to do soon

    Parameters
    ----------
    n_features : int, optional (default=20)
        The number of features.
    n_mindsets : int, optional (default=2)
        The number of classes, we call them "mindsets", that we should generate.

    Returns
    -------
    c : array of shape [n_mindsets, n_features]
        The coordinates of the blob_centers of the mindsets

    """

    next_power_two = np.int(np.power(2, np.ceil(np.log2(n_features))))
    h_matrix = hadamard(next_power_two)
    h_matrix = (h_matrix == 1)

    idxs = np.arange(next_power_two)
    idxs = np.random.choice(idxs, n_mindsets, replace=False)
    c = h_matrix[idxs, :n_features]

    return c


def make_binary_questionnaire(n_samples=100, n_features=20, n_mindsets=2, n_mistakes=2,
                              seed=None, centers=True):
    """

    This function simulates a synthetic questionnaire.

    The center of the mindsets will always disagree among each other in n_features / n_mindsets features.
    TODO: This is false, think how to fix it. I am not even sure it is possible to fix, it might be NP complete
          since it is equivalent to some reformulation of the clique problem.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of samples.
    n_features : int, optional (default=20)
        The number of features.
    n_mindsets : int, optional (default=2)
        The number of classes, we call them "mindsets", that we should generate.
    nb_mistakes: float, optional (default=0.2)
        The percentage of deviation, computed respected to the 0-1 loss, that we allow
         inside a mindset. Must be in [0,1].
    seed: Int, optional (default=None)
        If provided it sets the seed of the random generator
    centers : bool, optional (default=False)
        If True the function will return an additional array that contains the
        coordinates of the center of the mindset.

    Returns
    -------

    xs : array of shape [n_samples, n_features]
        The generated samples.
    ys : array of shape [n_samples]
        The integer labels for mindset membership of each sample.
    cs : array of shape [n_mindsets, n_features]
        The coordinates of the blob_centers of the mindsets

    """

    # TODO: We need some check on the number of mindsets vs number of samples and features.
    #       think how to implement that.

    if seed is not None:
        np.random.seed(seed)

    cs = make_centers(n_features, n_mindsets)
    xs, ys = np.empty((n_samples, n_features), dtype=np.bool), np.empty(n_samples, dtype=np.int)
    id_mindsets = np.array_split(np.arange(n_samples), n_mindsets)

    for (mindset, ids) in enumerate(id_mindsets):
        xs[ids, :] = cs[mindset]
        ys[ids] = mindset

    id_errors = np.random.choice(n_features, size=(n_samples, n_mistakes))
    xs_to_flip = xs[np.arange(len(id_errors)), id_errors.T]
    xs[np.arange(len(id_errors)), id_errors.T] = np.flip(xs_to_flip)

    if centers:
        return xs, ys, cs
    else:
        return xs, ys


def make_questionnaire(n_samples, n_features, n_mindsets, range_answers, seed=None):

    min_answer = range_answers[0]
    max_answer = range_answers[1]

    if seed is not None:
        np.random.seed(seed)

    xs, ys = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_mindsets)
    cs = []
    for i in np.unique(ys):
        xs_mindset = xs[ys == i]

        xs_mindset = np.interp(xs_mindset, (xs_mindset.min(), xs_mindset.max()), (min_answer, max_answer))
        c = np.floor(np.average(xs_mindset, axis=0)).astype(int)
        cs.append(c)
        xs[ys == i] = np.floor(xs_mindset).astype(int)

    for c in cs:
        print(c)

    xs = xs.astype(int)

    return xs, ys
