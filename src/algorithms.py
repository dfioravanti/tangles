import numpy as np
import bitarray as ba

from src.oriented_cuts import Specification


def core_algorithm(tangles, current_cuts, idx_current_cuts, min_size):

    last_added = len(tangles)

    for i, cut in zip(idx_current_cuts, current_cuts):
        current_tangles = []
        non_maximal = []

        if tangles == []:
            if np.sum(cut) >= min_size:
                array = ba.bitarray(list(cut))
                current_tangles.append(Specification([array], [array], {i: True}))
            if np.sum(~cut) >= min_size:
                array = ba.bitarray(list(~cut))
                current_tangles.append(Specification([array], [array], {i: False}))
        else:
            for i_tau, tau in enumerate(tangles[-last_added:]):
                tau_extended = False

                new_tangle = tau.add(ba.bitarray(list(cut)), {i: True}, min_size)
                if new_tangle is not None:
                    current_tangles.append(new_tangle)
                    tau_extended = True

                new_tangle = tau.add(ba.bitarray(list(~cut)), {i: False}, min_size)
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
        last_added = len(current_tangles)

    return tangles
