from copy import deepcopy

import numpy as np


def size(orientations, cuts):
    """
    Compute the size of a collection of oriented cuts

    Parameters
    ----------
    orientations: List of ndarray

    Returns
    -------
    size: Int
    """

    _, n = cuts.shape
    intersection = np.ones((1, n), dtype=bool)
    for i in orientations:
        current = cuts[i-1] if i >= 0 else ~cuts[-i-1]
        intersection = intersection * current

    size = np.sum(intersection)
    return size


def exponential_algorithm(xs, cuts):

    T = []
    threshold = np.trunc(len(xs) * 0.2)
    n_cuts = len(cuts)

    T_0 = []
    for orientation in [[1], [-1]]:
        if size(orientation, cuts) >= threshold:
            T_0.append(orientation)

    T.append(deepcopy(T_0))

    for i in range(1, n_cuts+1):
        T_i = []

        for tau in T[i-1]:
            for s_i in [[i+1], [-(i+1)]]:
                orientation = tau + s_i
                if size(orientation, cuts) >= threshold:
                    T_i.append(orientation)

        if len(T_i) == 0:
            break

        T.append(deepcopy(T_i))

    return T[-1]
