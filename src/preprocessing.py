from collections import deque

import numpy as np


def make_submodular(cuts):

    """
    Given a set of cuts we make it submodular.
    A set of cuts S is submodular if for any two orientation A,B of cuts in S we have that
    either A union B or A intersection B is in S.
    We achieve this by adding all the expressions composed by unions.
    The algorithm is explained in the paper.

    All the hashing stuff is necessary because numpy arrays are not hashable.

    # TODO: It does not scale up very well. We might need to rethink this

    Parameters
    ----------
    cuts: array of shape [n_cuts, n_users]
        The original cuts that we need to make submodular

    Returns
    -------
    new_cuts: array of shape [?, n_users]
        The submodular cuts
    """

    if len(cuts) == 1:
        return cuts

    unions = {}

    for current_cut in cuts:
        v = current_cut
        k = hash(v.tostring())
        current_unions = {k: v}

        for cut in unions.values():
            v = cut | current_cut
            k = hash(v.tostring())
            current_unions.setdefault(k, v)

            v = current_cut | ~cut
            k = hash(v.tostring())
            current_unions.setdefault(k, v)

            v = ~(~cut & current_cut)
            k = hash(v.tostring())
            current_unions.setdefault(k, v)

            v = ~(~cut & current_cut)
            k = hash(v.tostring())
            current_unions.setdefault(k, v)

        unions.update(current_unions)

    # Remove empty cut and all cut
    empty, all = np.zeros_like(current_cut, dtype=bool), np.ones_like(current_cut, dtype=bool)
    hash_empty, hash_all = hash(empty.tostring()), hash(all.tostring())
    unions.pop(hash_empty, None)
    unions.pop(hash_all, None)

    new_cuts = np.array(list(unions.values()), dtype='bool')

    return new_cuts


def find_comparable_components(cuts):

    _, n = cuts.shape
    idx = np.arange(n)
    components = []

    for cut in cuts:

        is_new_component = True
        for component in components:
            idx_biggest = set(idx[component[0]])
            idx_cut, idx_comp_cut = set(idx[cut]), set(idx[~cut])

            if idx_cut.issubset(idx_biggest):
                idx_smallest = set(idx[component[-1]])
                if idx_smallest.issubset(idx_cut):
                    component.append(cut)
                else:
                    for cut_component in component[1:-2]:
                        pass

            elif idx_biggest.issubset(idx_comp_cut):
                cut = ~cut
                component.appendleft(cut)

        if is_new_component:
            components.append(deque([cut]))

    return components
