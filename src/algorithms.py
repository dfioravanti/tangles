from copy import deepcopy

import numpy as np

from src.tangles import OrientedCut

def size(oriented_cuts, S):
    """
    Compute the size of a collection of oriented S

    Parameters
    ----------
    orientations: List of ndarray

    Returns
    -------
    size: Int
    """

    _, n = S.shape
    intersection = np.ones((1, n), dtype=bool)
    for cut, orientation in oriented_cuts:
        current = S[cut] if orientation else ~S[cut]
        intersection = intersection * current

    size = np.sum(intersection)
    return size


def exponential_algorithm(xs, cuts):

    T = []
    threshold = np.trunc(len(xs) * 0.2)
    n_cuts = len(cuts)

    T_0 = []
    for orientation in [True, False]:
        oriented_cuts = OrientedCut(cuts=[0], orientations=[orientation])
        if size(oriented_cuts, cuts) >= threshold:
            T_0.append(oriented_cuts)

    T.append(deepcopy(T_0))

    for i in range(1, n_cuts+1):
        T_i = []

        for tau in T[i-1]:
            for orientation in [True, False]:
                oriented_cuts = tau + OrientedCut(cuts=[i], orientations=[orientation])
                if size(oriented_cuts, cuts) >= threshold:
                    T_i.append(oriented_cuts)

        if len(T_i) == 0:
            break

        T.append(deepcopy(T_i))

    return T[-1]
