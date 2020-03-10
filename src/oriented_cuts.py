from itertools import combinations
from copy import deepcopy

from bitarray.util import subset


class Specification(dict):
    """
    This class represents an oriented cut as a couple of lists.
    In idx_cuts it is stored the index of the cut in S, meanwhile in orientation
    we store which orientation we are taking as a bool.
    """
    def __init__(self, values=[], core=[], idx={}):

        self.values = values
        self.core = core
        self.idx = idx

    def add(self, cut, new_idx, min_size):

        values = deepcopy(self.values)
        core = deepcopy(self.core)
        idx = deepcopy(self.idx)

        for i, core_cut in enumerate(core):
            if subset(core_cut, cut):
                values.append(cut)
                idx.update(new_idx)
                return Specification(values, core, idx)
            if subset(cut, core_cut):
                del core[i]

        if len(core) == 0:
            if cut.count() < min_size:
                return None
        elif len(core) == 1:
            if (core[0] & cut).count() < min_size:
                return None
        else:
            for core1, core2 in combinations(core, 2):
                if (core1 & core2 & cut).count() < min_size:
                    return None

        values.append(cut)
        core.append(cut)
        idx.update(new_idx)

        return Specification(values, core, idx)
