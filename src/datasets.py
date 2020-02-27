import numpy as np


def make_submodular(cuts):

    # TODO: ndarray is not hashable. Find a way around it

    unions = set(cuts[-1:])
    intersections = set(cuts[-1:])

    for current_cut in cuts[-2::-1]:
        current_cut = current_cut.tobytes()
        current_unions = set()
        current_intersections = set()

        for cut in unions:
            current_unions.add(np.logical_or(current_cut, cut))
            current_intersections.add(np.logical_and(current_cut, ~cut))

        for cut in intersections:
            current_unions.add(np.logical_or(current_cut, ~cut))
            current_intersections.add(np.logical_and(current_cut, cut))

        unions |= current_unions
        intersections |= current_intersections

    cuts = unions | intersections
    cuts = np.array(list(cuts))

    return cuts

