import numpy as np


class OrientedCut:
    """
    This class represents an oriented cut as a couple of lists.
    In cuts it is stored the index of the cut in S, meanwhile in orientation
    we store which orientation we are taking as a bool.
    """
    def __init__(self, cuts, orientations):

        self.current = -1

        if not isinstance(cuts, (list, np.ndarray)):
            cuts = [cuts]
            orientations = [orientations]

        self.cuts = np.array(cuts, dtype=int)
        self.orientations = np.array(orientations, dtype=bool)
        self.size = len(self.cuts)

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        if self.current < len(self.cuts):
            return self.cuts[self.current], self.orientations[self.current]
        raise StopIteration


def add_to_oriented_cut(oriented_cut, cut, orientation):
        i = np.searchsorted(oriented_cut.cuts, cut)
        if i < oriented_cut.size and oriented_cut.cuts[i] == cut and oriented_cut.orientations[i] == orientation:
            return oriented_cut, False
        else:
            new_cuts = np.insert(oriented_cut.cuts, i, cut)
            new_orientations = np.insert(oriented_cut.orientations, i, orientation)
            return OrientedCut(new_cuts, new_orientations), True
