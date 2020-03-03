from copy import deepcopy

from src.oriented_cuts import OrientedCut


def basic_algorithm(previous_tangles, current_cuts, acceptance_function):

    # TODO: Add the reuse of old tangles

    T = deepcopy(previous_tangles)
    last_added = len(T)

    for i in current_cuts:
        T_current = []
        non_maximal = []

        if len(T) == 0:
            for orientation in [True, False]:
                new_oriented_cut = OrientedCut({i: orientation})
                if acceptance_function(old_oriented_cuts=None, new_oriented_cut=new_oriented_cut):
                    T_current.append(new_oriented_cut)
        else:
            for i_tau, tau in enumerate(T[-last_added:]):
                tau_extended = False

                for orientation in [True, False]:
                    new_oriented_cut = OrientedCut({i: orientation})
                    if acceptance_function(old_oriented_cuts=tau, new_oriented_cut=new_oriented_cut):
                        T_current.append(tau + new_oriented_cut)
                        tau_extended = True

                if tau_extended:
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
