from copy import deepcopy

import numpy as np


class OrientedCut(dict):
    """
    This class represents an oriented cut as a couple of lists.
    In idx_cuts it is stored the index of the cut in S, meanwhile in orientation
    we store which orientation we are taking as a bool.
    """
    def __init__(self, oriented_cuts={}):

        super(OrientedCut, self).__init__(oriented_cuts)

    def __hash__(self):
        return hash(str(self.get_idx_cuts()) + str(self.get_orientations()))

    def __eq__(self, other):
        if self.get_idx_cuts() == other.get_idx_cuts() and self.get_orientations() == other.get_orientations():
            return True
        else:
            return False

    def __add__(self, other):
        return OrientedCut({**self, **other})

    def get_idx_cuts(self):
        return list(self.keys())

    def get_orientations(self):
        return list(self.values())

    def orientation_of(self, i):
        return self.get(i)

    def add(self, oriented_cut):
        return OrientedCut({**self, **oriented_cut})
