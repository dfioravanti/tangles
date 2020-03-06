from collections import deque


def triplet_size_big_enough(all_cuts, oriented_cuts, min_size):
    """
        This function checks if all triples in sum(oriented_cuts) have size at least min_size.
        We assume that all the triplets in a given cut have this property
        so we need to check only the new triplets.

        Parameters
        ----------
        all_cuts, array of shape [n_cuts, n_users]
            The matrix that contains the index for all the idx_cuts
        oriented_cuts, list of OrientedCut
            The list of oriented cuts that we need to test
        min_size, int
            The minimum number of points that we want to have in the intersection of triplets.

        Returns
        -------
        condition_satisfied, bool
            True if the property old, False otherwise

    """

    if not isinstance(oriented_cuts, list):
        return oriented_cuts.is_consistent(all_cuts, min_size)

    cuts = deque(oriented_cuts)
    for _ in range(len(cuts)):
        cut = cuts.popleft()
        for other_cut in cuts:
            if not cut.is_consistent_with(other_cut, all_cuts, min_size):
                return False

        cuts.append(cut)
    return True
