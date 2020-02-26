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
        oriented_cuts = OrientedCut(cuts=0, orientations=orientation)
        if size(oriented_cuts, cuts) >= threshold:
            T_0.append(oriented_cuts)

    T.append(deepcopy(T_0))

    for i in range(1, n_cuts+1):
        T_i = []
        non_maximal = []

        for j, tau in enumerate(T[i-1]):

            tau_extended = False

            for orientation in [True, False]:
                oriented_cuts = tau.add_oriented_cut(i, orientation)
                if size(oriented_cuts, cuts) >= threshold:
                    T_i.append(oriented_cuts)
                    tau_extended = True

            if tau_extended:
                non_maximal.append(j)

        if len(T_i) == 0:
            break

        for j in sorted(non_maximal, reverse=True):
            del T[i-1][j]

        T.append(deepcopy(T_i))

    T = list(filter(None, T))
    return T
