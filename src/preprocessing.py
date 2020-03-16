from itertools import combinations


import numpy as np
from sklearn.cluster import SpectralClustering

def make_submodular(cuts):

    """
    Given a set of cuts we make it submodular.
    A set of cuts S is submodular if for any two orientation A,B of cuts in S we have that
    either A union B or A intersection B is in S.
    We achieve this by adding all the expressions composed by unions.
    The algorithm is explained in the paper.

    All the hashing stuff is necessary because numpy arrays are not hashable.

    # TODO: It does not scale up very well. We might need to rethink this

    Parameters
    ----------
    cuts: array of shape [n_cuts, n_users]
        The original cuts that we need to make submodular

    Returns
    -------
    new_cuts: array of shape [?, n_users]
        The submodular cuts
    """

    if len(cuts) == 1:
        return cuts

    unions = {}

    for current_cut in cuts:
        v = current_cut
        k = hash(v.tostring())
        current_unions = {k: v}

        for cut in unions.values():
            v = cut | current_cut
            k = hash(v.tostring())
            current_unions.setdefault(k, v)

            v = current_cut | ~cut
            k = hash(v.tostring())
            current_unions.setdefault(k, v)

            v = ~(~cut & current_cut)
            k = hash(v.tostring())
            current_unions.setdefault(k, v)

            v = ~(~cut & current_cut)
            k = hash(v.tostring())
            current_unions.setdefault(k, v)

        unions.update(current_unions)

    # Remove empty cut and all cut
    empty, all = np.zeros_like(current_cut, dtype=bool), np.ones_like(current_cut, dtype=bool)
    hash_empty, hash_all = hash(empty.tostring()), hash(all.tostring())
    unions.pop(hash_empty, None)
    unions.pop(hash_all, None)

    new_cuts = np.array(list(unions.values()), dtype='bool')

    return new_cuts


def neighbours_in_same_cluster(idx_vertex, A, nb_common_neighbours):

    nb_verteces, _ = A.shape
    cut = np.zeros(nb_verteces, dtype=bool)

    neighbours = A[idx_vertex]
    idx_neighbours = np.where(neighbours == True)[0]
    for idx_neighbour in idx_neighbours:
        common_neighbours = np.sum(np.logical_and(neighbours, A[idx_neighbour]))
        if common_neighbours >= nb_common_neighbours:
            cut[idx_neighbour] = True

    return cut


def build_blob_graph(blobs):

    nb_blobs = len(blobs)
    A = np.zeros((nb_blobs, nb_blobs), dtype=int)
    idxs = np.triu_indices(nb_blobs, 1)

    for idx_1, idx_2, blob in zip(idxs[0], idxs[1], combinations(blobs, 2)):
        blob_1, blob_2 = blob
        common_points = np.sum(np.logical_and(blob_1, blob_2))
        A[idx_1, idx_2] = A[idx_2, idx_1] = common_points

    return A


def neighbourhood_cuts(A, nb_centers, nb_common_neighbours, max_k):

    nb_verteces, _ = A.shape
    idx = np.arange(nb_verteces)
    center_vertex = np.random.choice(idx, nb_centers, replace=False)

    blobs = np.empty(nb_verteces, bool)
    for idx_vertex in center_vertex:
        cut = neighbours_in_same_cluster(idx_vertex, A, nb_common_neighbours)
        if not(np.all(cut)) and np.any(cut):
            blobs = np.vstack([blobs, cut])

    blobs = blobs[1:]

    A_blobs = build_blob_graph(blobs)
    cuts = np.stack(blobs, axis=0)

    for k in range(2, max_k+1):
        cls = SpectralClustering(n_clusters=k, affinity='precomputed')
        clusters = cls.fit_predict(X=A_blobs)

        for cluster in range(0, k):
            current = blobs[clusters == cluster]
            current = np.any(current, axis=0)
            cuts = np.append(cuts, [current], axis=0)

    return cuts


if __name__ == '__main__':
    from src.datasets.loading.sbm import load_sbm

    A, ys = load_sbm(10, 4, .7, .3)
    neighbourhood_cuts(A, 10)
