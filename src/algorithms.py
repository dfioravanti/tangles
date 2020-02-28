from copy import deepcopy

import numpy as np

from src.tangles import OrientedCut, add_to_oriented_cut


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
        intersection = np.logical_and(intersection, current)

    size = np.sum(intersection)
    return size


def exponential_algorithm(xs, cuts):

    T = []
    threshold = np.trunc(len(xs) * 0.2)
    n_cuts = len(cuts)

    T_0 = {}

    for cut in np.arange(n_cuts):
        for orientation in [True, False]:
            oriented_cuts = OrientedCut(cuts=cut, orientations=orientation)
            if size(oriented_cuts, cuts) >= threshold:
                T_0[hash(oriented_cuts)] = oriented_cuts

    T.append(list(T_0.values()))
    i = 1

    while len(T[i-1]) != 0:
        T_i = {}
        non_maximal = []

        for i_tau, tau in enumerate(T[i-1]):
            tau_extended = False

            for cut in np.arange(n_cuts):
                for orientation in [True, False]:
                    oriented_cuts, changed = add_to_oriented_cut(tau, cut, orientation)
                    if changed and size(oriented_cuts, cuts) >= threshold:
                        T_i.setdefault(hash(oriented_cuts), oriented_cuts)
                        tau_extended = True

            if tau_extended:
                non_maximal.append(i_tau)

        for j in sorted(non_maximal, reverse=True):
            del T[i - 1][j]

        T.append(list(T_i.values()))

        i += 1

    T = [i for sub in T for i in sub]

    return T
