from itertools import combinations
from copy import deepcopy

from bitarray.util import subset


class Specification(dict):
    """
    This class represents an oriented cut as a couple of lists and a dictionary.
        - cuts contains all the biparitions of the specification defined as binary arrays.
          1 means that that x belongs to the partition and 0 that it does not.
          It is implemented with bitarrays for max speed
        - core contains all the biparitions of the core of the specification defined as binary arrays.
          1 means that that x belongs to the partition and 0 that it does not.
          It is implemented with bitarrays for max speed
        - specification is a dictionary there the key is the index of the cut in the list of all the cuts and
          the value is which orientation of that specification we need to take
    """
    def __init__(self, cuts=[], core=[], specification={}):

        """
        Initialise a new specification

        Parameters
        ----------
        cuts: list of bitarray
            All the biparitions of the specification
        core: list of bitarray
            All the biparitions of the core of the specification
        specification: dict of bool
            The key is the index of the cut in the list of all the cuts and
            the value is which orientation of that specification we need to take
        """

        self.cuts = cuts
        self.core = core
        self.specification = specification

    def add(self, new_cut, new_specification, min_size):

        """
        Check if new_cut can be added to the current specification

        Parameters
        ----------
        new_cut: bitarray
            The bipartition that we need to add as bitarray
        new_specification: dict of bool
            The orientation of new_cut
        min_size:
            Minimum triplet size that we accept for it to be a tangle

        Returns
        -------
        new_specification: Specification or None
            If it is possible to add we return the new specification otherwise we return None
        """

        cuts = deepcopy(self.cuts)
        core = deepcopy(self.core)
        specification = deepcopy(self.specification)

        for i, core_cut in enumerate(core):
            if subset(core_cut, new_cut):
                cuts.append(new_cut)
                specification.update(new_specification)
                return Specification(cuts, core, specification)
            if subset(new_cut, core_cut):
                del core[i]

        if len(core) == 0:
            if new_cut.count() < min_size:
                return None
        elif len(core) == 1:
            if (core[0] & new_cut).count() < min_size:
                return None
        else:
            for core1, core2 in combinations(core, 2):
                if (core1 & core2 & new_cut).count() < min_size:
                    return None

        cuts.append(new_cut)
        core.append(new_cut)
        specification.update(new_specification)

        return Specification(cuts, core, specification)
