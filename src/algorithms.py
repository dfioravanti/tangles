import numpy as np
import bitarray as ba

from src.oriented_cuts import Specification


def core_algorithm(tangles, current_cuts, idx_current_cuts, min_size):

    """
    Algorithm to compute the tangles by using the core heuristic.
    The algorithm can be outlined as follows
        1. For each old tangle we try to add the current cut in both orientations and
           we check if it can be done as outlined in the paper.
        2. If a tangle can be extended we mark it as extended, we remove the old one and add the new one


    Parameters
    ----------
    tangles: list
        The list of all FULL tangles computed up until now
    current_cuts: list
        The list of cuts that we need to try to add to the tangles. They are ordered by order
    idx_current_cuts: list
        The list of indexes of the current cuts in the list of all cuts
    min_size: int
        Minimum triplet size for a tangle to be a tangle.

    Returns
    -------
    tangles: list
        The list of all new tangles, both full and not
    """

    last_added = len(tangles)

    for i, cut in zip(idx_current_cuts, current_cuts):
        current_tangles = []
        non_maximal = []

        if tangles == []:
            if np.sum(cut) >= min_size:
                array = ba.bitarray(list(cut))
                current_tangles.append(Specification(cuts=[array],
                                                     core=[array],
                                                     specification={i: True})
                                       )
            if np.sum(~cut) >= min_size:
                array = ba.bitarray(list(~cut))
                current_tangles.append(Specification(cuts=[array],
                                                     core=[array],
                                                     specification={i: False})
                                       )
        else:
            for i_tau, tau in enumerate(tangles[-last_added:]):
                tau_extended = False

                new_tangle = tau.add(new_cut=ba.bitarray(list(cut)),
                                     new_specification={i: True},
                                     min_size=min_size)
                if new_tangle is not None:
                    current_tangles.append(new_tangle)
                    tau_extended = True

                new_tangle = tau.add(new_cut=ba.bitarray(list(~cut)),
                                     new_specification={i: False},
                                     min_size=min_size)
                if new_tangle is not None:
                    current_tangles.append(new_tangle)
                    tau_extended = True

                if tau_extended:
                    non_maximal.append(i_tau)

            len_tangles = len(tangles)
            for j in sorted(non_maximal, reverse=True):
                idx_remove = len_tangles - (last_added - j)
                del tangles[idx_remove]

        tangles += current_tangles
        last_added = len(current_tangles) if len(current_tangles) > 0 else last_added

    return tangles


def remove_incomplete_tangles(tangles, nb_cuts_considered):

    """
    Given a list of tangles we remove the tangles that are not using all the cuts available.
    In other words we remove all the partial specifications and we leave only the specifications

    Parameters
    ----------
    tangles: list
        The list of all tangles computed up until now
    nb_cuts_considered: int
        The number of cuts that have been considered up until now

    Returns
    -------
    tangles: list
        The list of all FULL tangles computed up until now
    """

    idx = reversed(range(len(tangles)))
    for i in idx:
        if len(tangles[i].cuts) < nb_cuts_considered:
            del tangles[i]

    return tangles
