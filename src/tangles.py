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

    def __add__(self, other):
        cuts = np.concatenate([self.cuts, other.cuts])
        orientations = np.concatenate([self.orientations, other.orientations])
        return OrientedCut(cuts, orientations)

    def add_oriented_cut(self, cut, orientation):
        return self + OrientedCut(cut, orientation)
