import numpy as np

from sklearn.metrics.pairwise import manhattan_distances


def min_center_distance(cs):
    ds = manhattan_distances(cs, cs).astype(np.int)
    not_diag_mask = ~np.eye(ds.shape[0], dtype=bool)

    return np.min(ds[not_diag_mask])
