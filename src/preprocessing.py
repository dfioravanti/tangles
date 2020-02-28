import numpy as np


def make_submodular(cuts):

    """
    Given a set of cuts we make it submodular. A set S is submodular if A union B and A intersection B are in S
    r all A, B in S.  We achieve this by adding all the binary expressions composed of only unions
    or intersections for every possible combination of elements of cuts. The algorithm is explained in the paper.

    All the hashing stuff is necessary because numpy arrays are not hashable.

    # TODO: It does not scale up very well. We might need to rethink this

    Parameters
    ----------
    cuts: ndarray
        The original cuts that we need to make submodular

    Returns
    -------
    new_cuts: ndarray
        The submodular cuts
    """

    current_cut = cuts[-1]
    hash_current_cut = hash(current_cut.tostring())
    unions = {hash_current_cut: current_cut}
    intersections = {hash_current_cut: current_cut}

    for current_cut in cuts[-2::-1]:
        current_unions, current_intersections = {}, {}

        for cut in unions.values():
            v = current_cut | cut
            k = hash(v.tostring())
            current_unions.setdefault(k, v)

            v = current_cut & ~cut
            k = hash(v.tostring())
            current_intersections.setdefault(k, v)

        for cut in intersections.values():
            v = current_cut | ~cut
            k = hash(v.tostring())
            current_unions.setdefault(k, v)

            v = current_cut & cut
            k = hash(v.tostring())
            current_intersections.setdefault(k, v)

        unions.update(current_unions)
        intersections.update(current_intersections)

    unions.update(intersections)
    new_cuts = np.array(list(unions.values()), dtype='bool')

    return new_cuts
