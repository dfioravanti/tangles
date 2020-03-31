from copy import deepcopy
from itertools import combinations
from random import sample

import numpy as np
from kmodes.kmodes import KModes
from sklearn.cluster import SpectralClustering
from sklearn.neighbors._dist_metrics import DistanceMetric

from src.config import PREPROCESSING_KARGER, PREPROCESSING_FAST_MINCUT


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


# ----------------------------------------------------------------------------------------------------------------------
# Kernighan-Lin algorithm
#
# implemented the pseudocode from https://en.wikipedia.org/wiki/Kernighan–Lin_algorithm
# also kept all the variable names
# it's a little difficult to see right away cause I often use indices
# might change to binary vectors, this could be much faster actually 
# ----------------------------------------------------------------------------------------------------------------------

def initial_partition(xs, fraction=0.5):
    nb_vertices, _ = xs.shape

    partition = round(fraction * nb_vertices)

    A = np.zeros(nb_vertices, dtype=bool)
    A[np.random.choice(np.arange(nb_vertices), partition, False)] = True

    B = np.logical_not(A)

    return A, B


def compute_D_values(xs, A, B):
    nb_vertices, _ = xs.shape

    D = np.zeros(nb_vertices)

    A_xs = xs[A, :]
    B_xs = xs[B, :]

    D[A] = np.sum(A_xs[:, B], axis=1) - np.sum(A_xs[:, A], axis=1)
    D[B] = np.sum(B_xs[:, A], axis=1) - np.sum(B_xs[:, B], axis=1)

    return D


def update_D_values(xs, A, B, D):
    nb_vertices, _ = xs.shape

    A_xs = xs[A, :]
    B_xs = xs[B, :]

    D[A] = np.sum(A_xs[:, B], axis=1) - np.sum(A_xs[:, A], axis=1)
    D[B] = np.sum(B_xs[:, A], axis=1) - np.sum(B_xs[:, B], axis=1)

    return D


def maximize(xs, A, B, D):
    g_max = -np.inf
    a_res = -1
    b_res = -1

    A_indices = A.nonzero()[0]
    B_indices = B.nonzero()[0]

    Dt = np.tile(D, (len(D), 1))
    G = Dt + Dt.T - 2 * xs
    G = G[A,:][:,B]
    (ai, bi) = np.unravel_index(np.argmax(G), G.shape)
    a_res = A_indices[ai]
    b_res = B_indices[bi]
    g_max = G[ai,bi]

    return g_max, a_res, b_res


def kernighan_lin(xs, nb_cuts, fractions):
    cuts = []

    for f in fractions:
        print(f"\t Calculating cuts for a fraction of: 1/{f}")
        for c in range(nb_cuts):
            cut = kernighan_lin_algorithm(xs, 1 / f)
            cuts.append(cut)

    cuts = np.array(cuts)

    return cuts

def kernighan_lin_algorithm(xs, fraction):
    nb_vertices, _ = xs.shape

    A, B = initial_partition(xs, fraction)

    while True:
        A_copy = A.copy()
        B_copy = B.copy()
        xs_copy = xs.copy()
        D = compute_D_values(xs, A_copy, B_copy)
        g_max = -np.inf
        g_acc = 0
        swap_max = np.empty_like(A)
        swap_acc = np.zeros_like(A)

        for _ in range(min(sum(A_copy), sum(B_copy))):
            # greedily find best two vertices to swap
            g, a, b = maximize(xs_copy, A_copy, B_copy, D)
            
            swap_acc[a] = True
            swap_acc[b] = True
            g_acc += g
            if g_acc > g_max:
                g_max = g_acc
                swap_max[:] = swap_acc[:]

            xs_copy[a, :] = 0
            xs_copy[:, a] = 0
            xs_copy[b, :] = 0
            xs_copy[:, b] = 0
            
            A_copy[a] = False
            B_copy[b] = False

            D = update_D_values(xs_copy, A_copy, B_copy, D)

        if g_max > 0:
            # swapping nodes from initial partition that improve the cut
            np.logical_not(A, out=A, where=swap_max)
            np.logical_not(A, out=B)

        else:
            break

    return A


# ----------------------------------------------------------------------------------------------------------------------
# Discrete approach
# ----------------------------------------------------------------------------------------------------------------------


def find_kmodes_cuts(xs, max_nb_clusters):
    nb_points = len(xs)
    cuts = []

    for k in range(2, max_nb_clusters):
        cls = KModes(n_clusters=k, init='Cao', n_init=1)
        clusters = cls.fit_predict(xs)
        print(f'Done with k={k}', flush=True)

        for cluster in range(0, k):
            cut = np.zeros(nb_points, dtype=bool)
            cut[clusters == cluster] = True
            if np.any(cut) and not np.all(cut):
                cuts.append(cut)

    cuts = np.stack(cuts, axis=0)
    return cuts


# ----------------------------------------------------------------------------------------------------------------------
# Graph approach
# ----------------------------------------------------------------------------------------------------------------------


def find_approximate_mincuts(A, nb_cuts, algorithm):
    """

    Generates a list of cuts by finding cheap approximations of min-cuts in a graph.

    Parameters
    ----------
     A: array of shape [nb_vertex, nb_vertex]
        The adjacency matrix of the graph
    nb_cuts: int
        The number of cuts to generate

    Returns
    -------
    cuts: array of shape [nb_cuts, nb_vertex]
        The cuts generated
    """

    cuts = []
    nb_vertices, _ = A.shape
    if algorithm == PREPROCESSING_KARGER:
        function = karger
    elif algorithm == PREPROCESSING_FAST_MINCUT:
        function = fast_min_cut
    else:
        raise Exception("Wrong algorithm name")

    while len(cuts) < nb_cuts:
        cut = np.zeros(nb_vertices, dtype=bool)
        idx_cuts, _, _ = function(A.astype(int))
        cut[idx_cuts] = True

        # at the moment sme hard coding to avoid super unbalanced cuts
        # does not make sense for more than 2 maybe 3 clusters and definitely needs to be changed later on
        if nb_vertices * 0.15 < sum(cut) < nb_vertices * 0.85:
            cuts.append(cut)

    cuts = np.array(cuts)

    return cuts


def karger(A):
    """

    Karger's randomized algorithm (https://en.wikipedia.org/wiki/Karger%27s_algorithm)
    Randomly choose an edge from the graph and shrink it by merging its adjacent vertices.
    Repeat until only 2 vertices are left. The two vertices represent the two sets of the separation.
    The algorithm finds a mincut with a probability of (1/n)^2.

    Parameters
    ----------
    A: array of shape [nb_vertex, nb_vertex]
        The adjacency matrix of the graph

    Returns
    -------
    separation_1: list of int
        list of indices representing the first separation
    separation_2: list of int
        list of indices representing the second separation
    weight_cut: int
        The weight of the merged cut

    """

    merged_nodes = []
    for i in range(len(A)):
        merged_nodes.append([i])

    A_shrunk, merged_nodes = contract(A, 2, merged_nodes)

    weight_cut = A_shrunk[0, 1]
    separation_1 = merged_nodes[0]
    separation_2 = merged_nodes[1]

    return separation_1, separation_2, weight_cut


def fast_min_cut(A, merged_nodes=None):
    if merged_nodes is None:
        merged_nodes = []
        for i in range(len(A)):
            merged_nodes.append([i])

    if len(A) <= 6:
        A_contracted, merged_nodes = contract(A, 2, merged_nodes)

        weight_cut = A_contracted[0, 1]
        separation_1 = merged_nodes[0]
        separation_2 = merged_nodes[1]
        return separation_1, separation_2, weight_cut
    else:
        min_size = np.int(np.ceil(1 + len(A) / np.sqrt(2)))
        A_1, merged_nodes1 = contract(A, min_size, merged_nodes)
        A_2, merged_nodes2 = contract(A, min_size, merged_nodes)

        separation_11, separation_12, cost_cut_1 = fast_min_cut(A_1, merged_nodes1)
        separation_21, separation_22, cost_cut_2 = fast_min_cut(A_2, merged_nodes2)

        if cost_cut_1 <= cost_cut_2:
            separation_1, separation_2 = separation_11, separation_12
            cost_cut = cost_cut_1
        else:
            separation_1, separation_2 = separation_21, separation_22
            cost_cut = cost_cut_2

    return separation_1, separation_2, cost_cut


def contract(A, min_size, merged_nodes):
    merged_nodes = deepcopy(merged_nodes)
    A_contracted = A.copy()

    while len(A_contracted) > min_size:
        i, j = pick_edge_from(A_contracted)
        A_contracted = merge(A_contracted, i, j)

        merged_nodes[i] += merged_nodes[j]
        del merged_nodes[j]

    return A_contracted, merged_nodes


def merge(A, i, j):
    """
    Merges two vertices i and j into one new vertex.
    The new vertex has as neighbours the union of the old  all the neighbours of the old.

    Parameters
    ----------
    A: array of shape [nb_vertex, nb_vertex]
        The adjacency matrix of the graph
    i: int
        The index of first vertex to merge
    j: int
        The index of second vertex to merge

    Returns
    -------
    A_merged: array of shape [nb_vertex - 1, nb_vertex - 1]
        The adjacency matrix of the shrunk graph
    """

    if max(i, j) > A.shape[0]:
        raise Exception("Graph already shrunk too much.")

    merged_vertex = A[i] + A[j]
    A[i] = merged_vertex
    A[:, i] = merged_vertex.T

    A[i, i] = 0
    A = np.delete(A, j, axis=0)
    A = np.delete(A, j, axis=1)

    return A


def pick_edge_from(A):
    """
    Pick a random edge from a graph and return the index of the adjacent vertices to this edge

    Parameters
    ----------
    A: array of shape [nb_vertex, nb_vertex]
        The adjacency matrix of the graph

    Returns
    -------
    i: int
        The index of first vertex to merge
    j: int
        The index of second vertex to merge
    """

    endpoints_1, endpoints_2 = np.nonzero(A)
    nb_edges = len(endpoints_1)
    edge = np.random.randint(nb_edges)

    i = endpoints_1[edge]
    j = endpoints_2[edge]

    if i < j:
        return i, j
    else:
        return j, i


# ----------------------------------------------------------------------------------------------------------------------
# Local approach
# ----------------------------------------------------------------------------------------------------------------------


def get_neighbours_few_edges(idx_neighbours, A):
    nb_edges = np.sum(A[idx_neighbours, :], axis=1)
    idxs = np.where(nb_edges <= 2)[0]

    return set(idxs)


def get_neighbour_patch(idx_vertex, A, min_size):
    nb_vertex = len(A)
    idx_patch = set()

    current_vertex = idx_vertex
    while len(idx_patch) <= min_size:
        idx_neighbours = list(np.where(A[current_vertex] > 0)[0])

        idx_patch.add(current_vertex)
        idx_patch = idx_patch.union(get_neighbours_few_edges(idx_neighbours, A))

        current_vertex = np.random.choice(idx_neighbours)

    patch = np.zeros(nb_vertex, dtype=bool)
    idx_patch = list(idx_patch)
    patch[idx_patch] = True

    return patch


def get_neighbour_cover(A, nb_cuts, percentages):
    cover = []
    nb_vertex = len(A)
    for percentage in percentages:
        min_size = nb_vertex * percentage
        for i in range(nb_cuts):
            idx_vertex = np.random.randint(low=0, high=nb_vertex)
            patch = get_neighbour_patch(idx_vertex, A, min_size)
            cover.append(patch)

    cover = np.stack(cover, axis=0)
    return cover


def get_random_cover(A, min_size_cover):
    nb_verteces, _ = A.shape
    idx_to_cover = set(range(nb_verteces))
    possible_centers = set(range(nb_verteces))
    cover = []
    hashes = set()
    size_cover = 0

    while len(possible_centers) != 0 and (size_cover < min_size_cover or len(idx_to_cover) > 0):
        idx_vertex = sample(possible_centers, 1)
        possible_centers -= set(idx_vertex)

        patch = A[idx_vertex].flatten()
        patch[idx_vertex] = True

        idx_patch = frozenset(np.where(patch == True)[0])
        hash_patch = hash(idx_patch)
        if hash_patch not in hashes:
            size_cover += 1
            hashes.add(hash_patch)
            cover.append(patch)
            idx_to_cover -= set(idx_patch)

    cover = np.stack(cover, axis=0)
    return cover


def random_cover_cuts(A, min_size_cover, dim_linspace):
    cover = get_random_cover(A, min_size_cover)
    dist = lambda xs, ys: np.sum(xs * ys)
    dist = DistanceMetric.get_metric(dist)
    nb_common_points = dist.pairwise(cover)

    max_nb_common_points = np.max(nb_common_points, axis=0)
    len_cover = len(cover)

    cuts = []
    hashes = set()

    fractions = np.linspace(start=0.1, stop=1, num=dim_linspace)
    for fraction in fractions:
        for idx_patch in np.arange(len_cover):
            close_patches = (nb_common_points[idx_patch] >= max_nb_common_points[idx_patch] * fraction)
            close_patches[idx_patch] = True
            patches = cover[close_patches, :]
            cut = np.sum(patches, axis=0).astype(bool)

            if np.sum(cut) >= 8 and np.sum(cut) <= 15:
                idx_cut = frozenset(np.where(cut == True)[0])
                hash_cut = hash(idx_cut)
                if hash_cut not in hashes:
                    hashes.add(hash_cut)
                    cuts.append(cut)

    cuts = np.stack(cuts, axis=0)
    return cuts


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
        cut = neighbours_in_same_cluster(vertex, A, nb_common_neighbours)

        cuts.append(cut)
        blob = set(np.where(cut == True)[0])
        cover.append(blob)

        idx_to_cover = idx_to_cover - set(blob)

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
