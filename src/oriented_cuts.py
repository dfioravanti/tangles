from itertools import combinations

import numpy as np


class OrientedCut(dict):
    """
    This class represents an oriented cut as a couple of lists.
    In idx_cuts it is stored the index of the cut in S, meanwhile in orientation
    we store which orientation we are taking as a bool.
    """
    def __init__(self, oriented_cuts={}, min_cuts=[]):

        super(OrientedCut, self).__init__(oriented_cuts)
        if len(oriented_cuts) == 1 and len(min_cuts) == 0:
            self.min_cuts = list(oriented_cuts.keys())
        else:
            self.min_cuts = min_cuts

    def __hash__(self):
        return hash(str(self.get_idx_cuts()) + str(self.get_orientations()))

    def __add__(self, other):

        for k, v in self.items():
            other_v = other.get(k)

            if other_v is not None and v != other_v:
                return None

        return OrientedCut({**self, **other})

    def get_min_cut(self):
        return self.min_cuts

    def get_idx_cuts(self):
        return list(self.keys())

    def get_orientations(self):
        return list(self.values())

    def orientation_of(self, i):
        return self.get(i)

    def get_partition(self, i, all_cuts):
        o = self.orientation_of(i)
        return all_cuts[i] if o else ~all_cuts[i]

    def add_superset(self, other):
        return OrientedCut(oriented_cuts={**self, **other}, min_cuts=self.min_cuts)

    def add_crossing (self, other):
        return OrientedCut(oriented_cuts={**self, **other}, min_cuts=self.min_cuts + other.min_cuts)

    def is_consistent(self, all_cuts, min_size):

        if len(self) == 0:
            return True

        if len(self.min_cuts) == 1:
            partition = self.get_partition(self.min_cuts[0], all_cuts)
            return False if np.sum(partition) < min_size else True

        if len(self.min_cuts) == 2:
            partition1 = self.get_partition(self.min_cuts[0], all_cuts)
            partition2 = self.get_partition(self.min_cuts[1], all_cuts)
            return False if np.sum(partition1 * partition2) < min_size else True

        if len(self) >= 3:
            for i1, i2, i3 in combinations(self.min_cuts, 3):
                partition1 = self.get_partition(i1, all_cuts)
                partition2 = self.get_partition(i2, all_cuts)
                partition3 = self.get_partition(i3, all_cuts)

                if np.sum(partition1 * partition2 * partition3) < min_size:
                    return False

            return True

    def is_consistent_with(self, other, all_cuts, min_size):

        if len(self) == 0:
            return True

        if len(self.min_cuts) == 1 and len(other.min_cuts) == 1:
            p_self = self.get_partition(self.min_cuts[0], all_cuts)
            p_other = other.get_partition(other.min_cuts[0], all_cuts)
            return False if np.sum(p_self * p_other) < min_size else True

        if len(self.min_cuts) == 2 and len(other.min_cuts) == 1:
            p1_self = self.get_partition(self.min_cuts[0], all_cuts)
            p2_self = self.get_partition(self.min_cuts[1], all_cuts)
            p_other = other.get_partition(other.min_cuts[0], all_cuts)
            return False if np.sum(p1_self * p2_self * p_other) < min_size else True

        if len(self.min_cuts) == 1 and len(other.min_cuts) == 2:
            p_self = self.get_partition(self.min_cuts[0], all_cuts)
            p1_other = other.get_partition(other.min_cuts[0], all_cuts)
            p2_other = other.get_partition(other.min_cuts[1], all_cuts)
            return False if np.sum(p_self * p1_other * p2_other) < min_size else True

        for i1_self, i2_self in combinations(self.min_cuts, 2):
            p1_self = self.get_partition(i1_self, all_cuts)
            p2_self = self.get_partition(i2_self, all_cuts)

            for i_other in other.min_cuts:
                p_other = other.get_partition(i_other, all_cuts)

                if np.sum(p1_self * p2_self * p_other) < min_size:
                    return False

        for i1_other, i2_other in combinations(other.min_cuts, 2):
            p1_other = self.get_partition(i1_other, all_cuts)
            p2_other = self.get_partition(i2_other, all_cuts)

            for i_self in self.min_cuts:
                p_self = other.get_partition(i_self, all_cuts)

                if np.sum(p_self * p1_other * p2_other) < min_size:
                    return False

        return True
