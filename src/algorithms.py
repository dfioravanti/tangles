from copy import deepcopy

from src.oriented_cuts import OrientedCut


def basic_algorithm(xs, cuts, previous_tangles, acceptance_function):

    # TODO: Add the reuse of old tangles

    T = deepcopy(previous_tangles)
    T_old = []

    for i in cuts:
        T_current = []
        non_maximal = []

        if len(T) == 0:
            for orientation in [True, False]:
                oriented_cuts = OrientedCut(cuts=i, orientations=orientation)
                if acceptance_function(oriented_cuts):
                    T_current.append(oriented_cuts)
        else:
            for i_tau, tau in enumerate(T_old):
                tau_extended = False

                for orientation in [True, False]:
                    oriented_cuts, _ = tau.add(i, orientation)
                    if acceptance_function(oriented_cuts):
                        T_current.append(oriented_cuts)
                        tau_extended = True

                if tau_extended:
                    non_maximal.append(i_tau)

            if len(T_current) == 0:
                raise Exception(f"I cannot find a tangle because cut {i} cannot be added")

            for j in sorted(non_maximal, reverse=True):
                del T[j]

        T += T_current
        T_old = T_current

    return T
