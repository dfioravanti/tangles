from itertools import combinations

import numpy as np


def triplet_size_big_enough(all_cuts, old_oriented_cuts, new_oriented_cut, min_size):
    """
    This function checks if all triples in old_old_orientations + new_oriented_cut have size
    at least min_size. We assume that all the triplets in old_orientations have this property
    so we need to check only the new triplets.

    Parameters
    ----------
    all_cuts, array of shape [n_cuts, n_users]
        The matrix that contains the index for all the idx_cuts
    old_oriented_cuts, OrientedCut
        The old orientations that we assume already have the property that we want
    new_oriented_cut, OrientedCut
        The new orientation that we want to add
    min_size, int
        The minimum number of points that we want to have in the intersection of triplets.

    Returns
    -------
    condition_satisfied, bool
        True if the property old, False otherwise

    """

    assert len(new_oriented_cut) > 0, "You cannot add an empty cut to an orientation"

    i, o = next(iter(new_oriented_cut.items()))
    new_orientation = all_cuts[i] if o else ~all_cuts[i]

    if old_oriented_cuts is None or len(old_oriented_cuts) == 0:

        if np.sum(new_orientation) >= min_size:
            condition_satisfied = True
        else:
            condition_satisfied = False

    elif len(old_oriented_cuts) == 1:

        i, o = next(iter(old_oriented_cuts.items()))
        old_orientation = all_cuts[i] if o else ~all_cuts[i]
        intersection = new_orientation * old_orientation

        if np.sum(intersection) >= min_size:
            condition_satisfied = True
        else:
            condition_satisfied = False

    elif len(old_oriented_cuts) == 2:

        old = iter(old_oriented_cuts.items())
        i, o = next(old)
        old_orientation1 = all_cuts[i] if o else ~all_cuts[i]
        i, o = next(old)
        old_orientation2 = all_cuts[i] if o else ~all_cuts[i]
        intersection = new_orientation * old_orientation1 * old_orientation2

        if np.sum(intersection) >= min_size:
            condition_satisfied = True
        else:
            condition_satisfied = False

    else:

        couples_old_idx = combinations(old_oriented_cuts.get_idx_cuts(), 2)
        condition_satisfied = True

        for old_i_1, old_i_2 in couples_old_idx:
            old_o_1 = old_oriented_cuts.orientation_of(old_i_1)
            old_orientation1 = all_cuts[old_i_1] if old_o_1 else ~all_cuts[old_i_1]
            old_o_2 = old_oriented_cuts.orientation_of(old_i_2)
            old_orientation2 = all_cuts[old_i_2] if old_o_2 else ~all_cuts[old_i_2]
            intersection = new_orientation * old_orientation1 * old_orientation2

            if np.sum(intersection) < min_size:
                condition_satisfied = False
                break

    return condition_satisfied
