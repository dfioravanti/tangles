
import numpy as np
from kmodes.kmodes import KModes

# ----------------------------------------------------------------------------------------------------------------------
# Kernighan-Lin algorithm
#
# implemented the pseudocode from https://en.wikipedia.org/wiki/Kernighan–Lin_algorithm
# also kept all the variable names
# it's a little difficult to see right away
# This is now fully vectorized. Everything still seems to work. Somebody should review this, though.
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
    Dtiled = np.tile(D, (len(D), 1))
    g = Dtiled + Dtiled.T - 2 * xs

    mask = np.logical_not(np.outer(A, B))
    g = np.ma.masked_array(g, mask)

    (a_res, b_res) = np.unravel_index(np.argmax(g), g.shape)
    g_max = g[a_res, b_res]

    return g_max, a_res, b_res


def kernighan_lin(xs, nb_cuts, fractions):
    cuts = []

    for f in fractions:
        # print(f"\t Calculating cuts for a fraction of: 1/{f}")
        for c in range(nb_cuts):
            cut = kernighan_lin_algorithm(xs, 0.5 + np.random.random() * .5)
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

# -----------------------------------------------
# Vertex-shifting local minimization approach
# -----------------------------------------------


def local_minimization(xs, nb_cuts):
    cuts = []
    xs = 1 * xs
    for _ in range(nb_cuts):
        cut = local_minimization_algorithm(xs)
        cuts.append(cut)

    return np.array(cuts)

def local_minimization_bounded(xs, nb_cuts):
    cuts = []
    xs = 1 * xs
    for _ in range(nb_cuts):
        cut = local_minimization_algorithm_bounded(xs)
        cuts.append(cut)

    return np.array(cuts)

def local_minimization_algorithm_bounded(xs):
    cutoff = 0.5 + np.random.rand() * 0.5
    cut, _ = initial_partition(xs, .5)
    n, _ = xs.shape
    domain = np.arange(n)
    cutvalue = cut @ xs @ (~cut)
    while True:
        if cut.sum() < n / 2:
            np.logical_not(cut, out=cut)
        np.random.shuffle(domain)
        for i in domain:
            newcut = cut.copy()
            newcut[i] = not newcut[i]
            if newcut.sum() > cutoff * n:
                continue
            newvalue = newcut @ xs @ (~newcut)
            if newvalue < cutvalue or (newvalue == cutvalue and cut[i] and cut.sum() > (n+1)/2 ) :
                cut = newcut
                cutvalue = newvalue
                break
        else:
            break
            
    return cut

def local_minimization_algorithm(xs):
    cut, _ = initial_partition(xs, np.random.rand())
    n, _ = xs.shape
    domain = np.arange(n)
    cutvalue = cut @ xs @ (~cut)
    
    optimal = False
    while not optimal:
        if cut.sum() <= n // 2:
            np.logical_not(cut, out=cut)
        np.random.shuffle(domain)
        optimal = True
        for side in [True, False]:
            if not optimal:
                break
            for i in domain:
                if cut[i] != side:
                    continue
                newcut = cut.copy()
                newcut[i] = not newcut[i]
                newvalue = newcut @ xs @ (~newcut)
                if newvalue < cutvalue:
                    cut = newcut
                    cutvalue = newvalue
                    optimal = False
                    break
            
            
    return cut



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
# Kneip fundamental cuts
# ----------------------------------------------------------------------------------------------------------------------
import networkx as nx

def kneip(adj, nb_cuts):
    G = nx.from_numpy_array(adj)
    cuts = np.zeros((0, len(G)), dtype=bool)
    while len(cuts) < nb_cuts:
        cuts = np.append(cuts, all_karger_fundamental_cuts(G), axis=0)
        cuts = np.unique(cuts, axis=0)
        print(f'\t\t I\'ve found {len(cuts)} cuts so far')
    return cuts

def all_karger_fundamental_cuts(G):
    edges = list(G.edges)
    np.random.shuffle(edges)
    for i, e in enumerate(edges):
        G.edges[e]['weight'] = i
    
    T = nx.minimum_spanning_tree(G, algorithm='kruskal') # May be a forest
    while not nx.is_connected(T):
        CC = list(nx.connected_components(T))
        C1, C2 = np.random.choice(CC, 2, replace=False)
        v1 = np.random.choice(list(C1))
        v2 = np.random.choice(list(C2))
        T.add_edge(v1,v2)
    return fundamental_cuts(G, T)

def fundamental_cuts(G, T):
    cuts = []
    for e in T.edges:
        T2 = T.copy()
        T2.remove_edge(*e)
        A, B = nx.connected_components(T2)
        acceptance = (2 * min(len(A), len(B)) / len(G))
        if np.random.rand() > acceptance:
            continue
        cut = np.zeros(len(T), dtype=bool)
        for a in A:
            cut[a] = True
        cuts.append(cut)
    return cuts

