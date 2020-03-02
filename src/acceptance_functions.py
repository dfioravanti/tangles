import numpy as np


def size_too_small(all_cuts, min_size, oriented_cuts):

    if size(oriented_cuts, all_cuts) >= min_size:
        return False
    else:
        return True


def size(oriented_cuts, all_cuts):
    """
    Compute the size of a collection of oriented all_cuts

    Parameters
    ----------
    orientations: List of ndarray

    Returns
    -------
    size: Int
    """

    _, n = all_cuts.shape
    intersection = np.ones((1, n), dtype=bool)
    for cut, orientation in oriented_cuts:
        current = all_cuts[cut] if orientation else ~all_cuts[cut]
        intersection = np.logical_and(intersection, current)

    size = np.sum(intersection)
    return size
