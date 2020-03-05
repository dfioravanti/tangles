from itertools import combinations

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

    def __add__(self, other):

        for k, v in self.items():
            other_v = other.get(k)

            if other_v is not None and v != other_v:
                return None

        return OrientedCut({**self, **other})

    def get_idx_cuts(self):
        return list(self.keys())

    def get_orientations(self):
        return list(self.values())

    def orientation_of(self, i):
        return self.get(i)

    def get_partition(self, i, all_cuts):
        o = self.orientation_of(i)
        return all_cuts[i] if o else ~all_cuts[i]

    def is_consistent(self, all_cuts, min_size):

        if len(self) == 0:
            return True

        if len(self) == 1:
            partition = self.get_partition(self.get_idx_cuts()[0], all_cuts)
            return False if np.sum(partition) < min_size else True

        if len(self) == 2:
            partition1 = self.get_partition(self.get_idx_cuts()[0], all_cuts)
            partition2 = self.get_partition(self.get_idx_cuts()[1], all_cuts)
            return False if np.sum(partition1 * partition2) < min_size else True

        if len(self) >= 3:
            for i1, i2, i3 in combinations(self.get_idx_cuts(), 3):
                partition1 = self.get_partition(i1, all_cuts)
                partition2 = self.get_partition(i2, all_cuts)
                partition3 = self.get_partition(i3, all_cuts)

                if np.sum(partition1 * partition2 * partition3) < min_size:
                    return False

            return True

    def is_consistent_with(self, new_i, new_o, all_cuts, min_size):

        if len(self) == 0:
            return True

        new_partition = all_cuts[new_i] if new_o else ~all_cuts[new_i]
        if len(self) == 1:
            partition = self.get_partition(self.get_idx_cuts()[0], all_cuts)
            return False if np.sum(partition * new_partition) < min_size else True

        if len(self) >= 2:
            for i1, i2 in combinations(self.get_idx_cuts(), 2):
                partition1 = self.get_partition(i1, all_cuts)
                partition2 = self.get_partition(i2, all_cuts)

                if np.sum(partition1 * partition2 * new_partition) < min_size:
                    return False

            return True

    def add(self, oriented_cut):
        return OrientedCut({**self, **oriented_cut})
