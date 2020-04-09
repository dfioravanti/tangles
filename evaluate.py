import numpy as np

def wcut(x, A, w=None):
    """ Computes weighted ratio cut value for a given flat vector x."""
    if w is None:
        w = np.ones(A.shape[0])
    return (1 / (x @ w) + 1 / ((1 - x) @ w)) * (x @ A) @ (1 - x)

def evaluate_SBM_partition(x, sizes):
    """ Evaluate the fractions that are separated by a partition in an SBM."""
    fractions = []
    pos = 0
    for i in range(len(sizes)):
        fractions.append(x[pos: pos + sizes[i]].mean())
        pos += sizes[i]
    return fractions

def evaluate_regions(node_sets, sizes):
    fractions = []
    for s in node_sets:
        fractions_s = []
        pos = 0
        for i in range(len(sizes)):
            block = set(np.arange(pos, pos+sizes[i]))
            fractions_s.append(len(block.intersection(s)) / len(s))
            pos += sizes[i]
        fractions.append(fractions_s)
    return fractions

# def evaluate_regions(node_sets, sizes):
#     fractions = []
#     for s in node_sets:
#         fractions_s = []
#         pos = 0
#         for i in range(len(sizes)):
#             block = set(np.arange(pos, pos+sizes[i]))
#             fractions_s.append(len(block.intersection(s)) / sizes[i])
#             pos += sizes[i]
#         fractions.append(fractions_s)
#     return fractions