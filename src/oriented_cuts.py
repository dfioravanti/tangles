from copy import deepcopy

import numpy as np


class OrientedCut:
    """
    This class represents an oriented cut as a couple of lists.
    In idx_cuts it is stored the index of the cut in S, meanwhile in orientation
    we store which orientation we are taking as a bool.
    """
    def __init__(self, oriented_cuts={}):

        self.current = -1
        self.oriented_cuts = oriented_cuts
        self.size = len(self.oriented_cuts)

    def __hash__(self):
        return hash(str(self.get_idx_cuts()) + str(self.get_orientations()))

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        if self.current == self.size:
            raise StopIteration

        cut = list(self.oriented_cuts.keys())[self.current]
        orientation = self.oriented_cuts[cut]
        return cut, orientation

    def __eq__(self, other):
        if self.get_idx_cuts() == other.get_idx_cuts() and self.get_orientations() == other.get_orientations():
            return True
        else:
            return False

    def __add__(self, other):
        return OrientedCut({**self.oriented_cuts, **other.oriented_cuts})

    def get_idx_cuts(self):
        return list(self.oriented_cuts.keys())

    def get_orientations(self):
        return list(self.oriented_cuts.values())

    def orientation_of(self, i):
        return self.oriented_cuts.get(i)

    def add(self, oriented_cut):
        return OrientedCut({**self.oriented_cuts, **oriented_cut})
