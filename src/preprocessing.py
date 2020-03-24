from itertools import combinations
from random import sample

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

    """
    Compute a local patch for the current vertex v.
    Give a point we check which of its neighbours u share at least nb_common_neighbours with v.
    If u has such property then we add it to the same patch as v. Otherwise we do not.

    Parameters
    ----------
    idx_vertex: int
        index of the current vertex in the adjacency matrix
    A: array of shape [nb_vertices, nb_vertices]
        The adjacency matrix for the graph
    nb_common_neighbours: int
        Minimal number of common neighbours that the neighbours of the current vertex need to share

    Returns
    -------

    """

    nb_verteces, _ = A.shape
    cut = np.zeros(nb_verteces, dtype=bool)

    cut[idx_vertex] = True
    neighbours = A[idx_vertex, :].flatten()
    idx_neighbours = np.where(neighbours == True)[0]
    for idx_neighbour in idx_neighbours:
        common_neighbours = np.sum(np.logical_and(neighbours, A[idx_neighbour]))
        is_only_neighbour = (A[idx_neighbour].sum() == 1 and A[idx_vertex, idx_neighbour] == True)
        if common_neighbours >= nb_common_neighbours or is_only_neighbour:
            cut[idx_neighbour] = True

    return cut


def build_cover_graph(cover, A):

    """
    Given a graph and a cover we build a new graph where the the vertex are the elements of the cover and the edges are
    weighted on the number of points connecting two different elements of the cover.

    Parameters
    ----------
    cover: list
        A cover for the graph
    A: array of shape [nb_vertices, nb_vertices]
       The adjacency matrix for the original graph

    Returns
    -------
    A_cover: array of shape [len_cover, len_cover]
       The adjacency matrix for the cover graph
    """

    size_cover = len(cover)
    A_cover = np.zeros((size_cover, size_cover), dtype=int)
    idxs = np.triu_indices(size_cover, 1)

    for idx_1, idx_2, blob in zip(idxs[0], idxs[1], combinations(cover, 2)):
        blob_1, blob_2 = blob
        blob_2 = blob_2 - blob_1
        ixgrid = np.ix_(np.array(list(blob_1)), np.array(list(blob_2)))

        if len(blob_2) == 0:
            nb_connecting_edges = len(A)
        else:
            nb_connecting_edges = np.sum(A[ixgrid])

        A_cover[idx_1, idx_2] = A_cover[idx_2, idx_1] = nb_connecting_edges

    return A_cover


def cuts_from_neighbourhood_cover(A, nb_common_neighbours, max_k):

    """
    Give a graph we we use a neighbourhood cover to decide which cuts we should use in the tangle algorithm.
    Such cover is built in two steps
        1. We sample a random vertex, we compute the neighbours cover of such vertex, we remove the cover from
           the pool of remaining points and then we keep going until we finish the points
        2. We use a clustering algorithm to cluster the cover that we found before, for each cluster we join all the vertex
           in that cluster into one cut and then we add such cut to our selection of cuts

    Parameters
    ----------
    A: array of shape [nb_vertices, nb_vertices]
       The adjacency matrix for the original graph
    nb_common_neighbours: int
        Minimal number of common neighbours that we use to build the cover
    max_k: int
        Maximum number of clusters that we look for in step 2

    Returns
    -------
    new_cuts: array of shape [nb_cuts, nb_users]
        The cuts
    """

    nb_verteces, _ = A.shape
    idx_to_cover = set(range(nb_verteces))
    cover = []
    cuts = []

    while len(idx_to_cover) > 0:

        vertex = sample(idx_to_cover, 1)
        cut = A[vertex].astype(bool)
        cut = np.asarray(cut).flatten()
        cut[vertex] = True

        cuts.append(cut)
        blob = set(np.where(cut == True)[0])
        cover.append(blob)

        idx_to_cover = idx_to_cover - blob

    initial_cuts = np.stack(cuts, axis=0)
    A_cover = build_cover_graph(cover, A)
    cuts = initial_cuts.copy()
    cover = np.array(cover)
    max_k = min(max_k, len(cover))

    for k in range(2, max_k):
        cls = SpectralClustering(n_clusters=k, affinity='precomputed')
        clusters = cls.fit_predict(X=A_cover)

        for cluster in range(0, k):
            cuts_in_cluster = initial_cuts[clusters == cluster]
            cut = np.any(cuts_in_cluster, axis=0)
            if np.any(cut) and not np.all(cut):
                cuts = np.append(cuts, [cut], axis=0)

    return cuts
