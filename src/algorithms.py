from copy import deepcopy

import numpy as np

from src.oriented_cuts import OrientedCut, get_partition, add_subset, add_superset, add_cross, partition_to_set


def core_algorithm(previous_tangles, current_cuts, all_cuts, min_size):

    """

    Parameters
    ----------
    previous_tangles
    current_cuts
    acceptance_function

    Returns
    -------

    """

    T = deepcopy(previous_tangles)
    last_added = len(T)

    for i in current_cuts:
        T_current = []
        non_maximal = []

        if len(T) == 0:
            for orientation in [True, False]:
                new_oriented_cut = OrientedCut({i: orientation})

                if new_oriented_cut.is_consistent(all_cuts, min_size):
                    T_current.append(new_oriented_cut)
        else:
            for i_tau, tau in enumerate(T[-last_added:]):
                is_tau_extended = False

                for orientation in [True, False]:
                    new_oriented_cut = OrientedCut({i: orientation})
                    tau_extended = add_new_cut(tau, new_oriented_cut, all_cuts, min_size)
                    if tau_extended is not None:
                        T_current.append(tau_extended)
                        is_tau_extended = True

                non_maximal.append(i_tau)

            len_T = len(T)
            for j in sorted(non_maximal, reverse=True):
                idx_remove = len_T - (last_added - j)
                del T[idx_remove]

        # A tangle has to orient every cut. If I cannot orient cut i then there is no tangle
        if len(T_current) == 0:
            return []

        T += T_current
        last_added = len(T_current)

    return T


def add_new_cut(partial_tangle, oriented_cut, all_cuts, min_size):

    i_new, o_new = next(iter(oriented_cut.items()))
    new_partition = get_partition(i_new, o_new, all_cuts)
    new_partition_idx = partition_to_set(new_partition)
    core_partitions = partial_tangle.get_core_partitions(all_cuts)

    for pos_core, core_partition in enumerate(core_partitions):

        core_partition_idx = partition_to_set(core_partition)
        if new_partition_idx.issuperset(core_partition_idx):
            return add_superset(partial_tangle, oriented_cut)
        if new_partition_idx.issubset(core_partition_idx):
            temp_core = np.delete(core_partitions, pos_core, axis=0)
            new_partial_tangle = add_subset(partial_tangle, oriented_cut, temp_core, new_partition, min_size)
            return new_partial_tangle

    new_partial_tangle = add_cross(partial_tangle, oriented_cut, core_partitions, new_partition, min_size)
    return new_partial_tangle
