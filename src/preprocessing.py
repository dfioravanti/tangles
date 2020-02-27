import numpy as np
from bitarray import bitarray


def make_submodular(cuts):

    # TODO: ndarray is not hashable. Find a way around it

    unions, intersections = set(), set()
    unions.add(tuple(cuts[-1]))
    intersections.add(tuple(cuts[-1]))

    for current_cut in cuts[-2::-1]:
        current_cut = tuple(current_cut)
        current_unions = set()
        current_intersections = set()

        for cut in unions:
            cut = np.array(cut, dtype='bool')
            current_unions.add(tuple(current_cut | cut))
            current_intersections.add(tuple(current_cut & ~cut))

        for cut in intersections:
            cut = np.array(cut, dtype='bool')
            current_unions.add(tuple(current_cut | ~cut))
            current_intersections.add(tuple(current_cut & cut))

        unions |= current_unions
        intersections |= current_intersections

    cuts = unions | intersections
    cuts = np.array(list(cuts), dtype='bool')
    return cuts
