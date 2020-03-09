from itertools import permutations
from src.oriented_cuts import Specification, add_cross, add_subset, add_supset, merge_cross


def find_tangles(C, k):
    # Leaf case
    if len(C.subsets) == 0:
        if len(C.partition) >= k:
            sigma_plus = [Specification([C.partition], [C.partition])]
        else:
            sigma_plus = []
        if len(C.complement_partition) >= k:
            sigma_minus = [Specification([C.complement_partition], [C.complement_partition])]
        else:
            sigma_minus = []
        return sigma_plus, sigma_minus

    # Not leaf case
    sigma_plus_child, sigma_minus_child = [], []
    node = C.subsets
    while node is not None:
        plus, minus = find_tangles(node, k)
        sigma_plus_child += plus
        sigma_minus_child += minus

        sigma_plus, sigma_minus = [], []
        for sigmas in permutations(sigma_plus_child):
            sigma = merge_cross(sigmas, k)
            if sigma is not None: sigma_plus.append(add_supset(node.partition, sigma))

        for sigmas in permutations(sigma_minus_child):
            sigma = merge_cross(sigmas, k)
            if sigma is not None: sigma_plus.append(add_cross(node.partition, sigma, k))
            if sigma is not None: sigma_plus.append(add_subset(node.partition, sigma, k))

        node = node.components

    return sigma_plus, sigma_minus
