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