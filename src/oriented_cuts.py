from itertools import combinations

from copy import deepcopy

import numpy as np


class OrientedCut(dict):
    """
    This class represents an oriented cut as a couple of lists.
    In idx_cuts it is stored the index of the cut in S, meanwhile in orientation
    we store which orientation we are taking as a bool.
    """
    def __init__(self, orientations={}, core=[]):

        super(OrientedCut, self).__init__(orientations)
        if len(orientations) == 1 and len(core) == 0:
            self.core = list(orientations.keys())
        else:
            self.core = core

    def __hash__(self):
        return hash(str(self.get_idx_cuts()) + str(self.get_orientations()))

    def get_idx_cuts(self):
        return list(self.keys())

    def get_orientations(self):
        return list(self.values())

    def orientation_of(self, i):
        return self.get(i)

    def get_partition_mask(self, i, all_cuts):
        o = self.orientation_of(i)
        return all_cuts[i] if o else ~all_cuts[i]

    def get_partitions_mask(self, all_cuts):
        for i, o in self.items():
            yield all_cuts[i] if o else ~all_cuts[i]

    def get_core_partitions_mask(self, all_cuts):
        for i in self.core:
            o = self.orientation_of(i)
            yield all_cuts[i] if o else ~all_cuts[i]

    def get_core_partitions(self, all_cuts):
        _, n = all_cuts.shape
        size = len(self.core)
        partitions = np.zeros((size, n), dtype=bool)

        for pos, i in enumerate(self.core):
            o = self.orientation_of(i)
            partitions[pos, :] = all_cuts[i] if o else ~all_cuts[i]

        return partitions

    def is_consistent(self, all_cuts, min_size):

        if len(self) == 0:
            return True

        i = self.core[0]
        o = self.orientation_of(i)
        partition = all_cuts[i] == o
        return False if np.sum(partition) < min_size else True


def is_core_consistent_with(core_partitions, new_partition, min_size):

    if len(core_partitions) == 0:
        return True

    if len(core_partitions) == 1:
        return False if np.sum(core_partitions[0] * new_partition) < min_size else True

    if len(core_partitions) == 2:
        return False if np.sum(core_partitions[0] * core_partitions[1] * new_partition) < min_size else True

    for couple_core_partition in combinations(core_partitions, 2):
        if np.sum(couple_core_partition[0] * couple_core_partition[1] * new_partition) < min_size:
            return False

    return True


def add_superset(cut1, cut2):
    return OrientedCut(orientations={**cut1, **cut2}, core=cut1.core)


def add_subset(partial_tangle, oriented_cut, core_partitions, new_partition, min_size):
    if is_core_consistent_with(core_partitions, new_partition, min_size):
        new_core = deepcopy(partial_tangle.core)
        new_core += oriented_cut.core
        return OrientedCut(orientations={**partial_tangle, **oriented_cut}, core=new_core)
    else:
        return None


def add_cross(partial_tangle, oriented_cut, core_partitions, new_partition, min_size):
    if is_core_consistent_with(core_partitions, new_partition, min_size):
        return OrientedCut(orientations={**partial_tangle, **oriented_cut},
                           core=partial_tangle.core + oriented_cut.core)
    else:
        return None


def get_partition(i, o, all_cuts):
    return all_cuts[i] == o


def partition_to_set(partition):
    idx = np.where(partition == True)[0]
    return set(idx.tolist())
