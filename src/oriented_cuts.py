from itertools import combinations

from copy import deepcopy

import numpy as np


class Specification(dict):
    """
    This class represents an oriented cut as a couple of lists.
    In idx_cuts it is stored the index of the cut in S, meanwhile in orientation
    we store which orientation we are taking as a bool.
    """
    def __init__(self, values=[], core=[]):

        self.values = values
        self.core = core


def add_supset(new_partition, specification):

    new_spec = deepcopy(specification)
    new_spec.values += [new_partition]

    return new_spec


def add_subset(new_partition, specification, min_size):

    new_spec = deepcopy(specification)
    new_spec.core.pop()

    for idx1, idx2 in combinations(specification.core, 2):
        core_partition1, core_partition2 = specification.values[idx1], specification.values[idx2]
        if len(new_partition.intersection(core_partition1, core_partition2)) < min_size:
            return None

    new_spec.values.append(new_partition)
    new_spec.core.append(new_partition)

    return new_spec


def add_cross(new_partition, specification, min_size):

    new_spec = deepcopy(specification)
    partition_to_replace = None

    for core_partition1, core_partition2 in combinations(specification.core, 2):

        if core_partition1.issubset(new_partition) or core_partition2.issubset(new_partition):
            new_spec.values.append(new_partition)
            return new_spec

        if len(new_partition.intersection(core_partition1, core_partition2)) < min_size:
            return None

        if new_partition.issubset(core_partition1):
            partition_to_replace = core_partition1
        elif new_partition.issubset(core_partition2):
            partition_to_replace = core_partition2

    new_spec.core.delete(partition_to_replace)
    new_spec.values.append(new_partition)
    new_spec.core.append(new_partition)

    return new_spec


def merge_cross(specifications, min_size):

    core = [spec.core for spec in specifications]

    for core_partition1, core_partition2, core_partition3 in combinations(*core, 3):
        if len(core_partition1.intersection(core_partition2, core_partition3)) < min_size:
            return None

    core = sum(core, [])
    values = sum([spec.values for spec in specifications], [])

    new_spec = Specification(core, values)
    return new_spec
