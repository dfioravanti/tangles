from copy import deepcopy

import numpy as np


class OrientedCut:
    """
    This class represents an oriented cut as a couple of lists.
    In idx_cuts it is stored the index of the cut in S, meanwhile in orientation
    we store which orientation we are taking as a bool.
    """
    def __init__(self, idx_cuts=[], orientations=[]):

        self.current = -1
        self._oriented_cuts = {}
        if not idx_cuts == []:
            if not isinstance(idx_cuts, (list, np.ndarray)):
                self._oriented_cuts[idx_cuts] = orientations
            else:
                for c, o in zip(idx_cuts, orientations):
                    if c not in self._oriented_cuts:
                        self._oriented_cuts[c] = o
                    else:
                        raise KeyError(f"cut {c} already present in the orientation")

        self.size = len(self._oriented_cuts)

    def __hash__(self):
        return hash(str(self.get_idx_cuts()) + str(self.get_orientations()))

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        if self.current == self.size:
            raise StopIteration

        cut = list(self._oriented_cuts.keys())[self.current]
        orientation = self._oriented_cuts[cut]
        return cut, orientation

    def __eq__(self, other):
        if self.get_idx_cuts() == other.get_idx_cuts() and self.get_orientations() == other.get_orientations():
            return True
        else:
            return False

    def get_idx_cuts(self):
        return list(self._oriented_cuts.keys())

    def get_orientations(self):
        return list(self._oriented_cuts.values())

    def orientation_of(self, i):
        return self._oriented_cuts.get(i)

    def add(self, cut, orientation):
        if cut in self._oriented_cuts:
            return self, False
        else:
            new_cut = deepcopy(self)
            new_cut._oriented_cuts[cut] = orientation
            new_cut.size += 1
            return new_cut, True
