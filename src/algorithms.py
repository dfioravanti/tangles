from itertools import product
from copy import deepcopy

import numpy as np

from src.oriented_cuts import OrientedCut
from src.components_tree import Node


def basic_algorithm(previous_tangles, current_cuts, acceptance_function):

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


def get_tangles(node, acceptance_function):

    T, t_components = [], []

    while node is not None:

        t_current_node, t_children = [], []

        child = node.subsets
        while child is not None:

            t_children += get_tangles(child, acceptance_function)
            child = child.incomps

        cut = OrientedCut({node.i_cut: True})
        comp_cut = OrientedCut({node.i_cut: False})

        if t_children:
            for tau in t_children:
                if acceptance_function(oriented_cuts=[tau, cut]):
                    t_current_node.append(tau + cut)
                if acceptance_function(oriented_cuts=[tau, comp_cut]):
                    t_current_node.append(tau + comp_cut)
        else:
            if acceptance_function(oriented_cuts=cut):
                t_current_node.append(cut)
            if acceptance_function(oriented_cuts=comp_cut):
                t_current_node.append(comp_cut)

        t_components.append(t_current_node)
        node = node.incomps

    for combination in product(*t_components):
        tangle = OrientedCut()
        for tau in combination:
            tangle += tau
            if tangle is None:
                break

        if tangle is not None and acceptance_function(oriented_cuts=tangle):
            T.append(tangle)

    return T


def tree_components_algorithm(idx_cuts, all_cuts, acceptance_function):

    tree = Node(all_cuts[idx_cuts[0]], idx_cuts[0])
    for i_cut in idx_cuts[1:]:
        tree, flip_cut = tree.insert(all_cuts[i_cut], i_cut)
        if flip_cut:
            all_cuts[i_cut] = ~all_cuts[i_cut]

    tangles = get_tangles(tree, acceptance_function)
    return tangles
