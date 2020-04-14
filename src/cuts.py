import numpy as np
from kmodes.kmodes import KModes

import src.coarsening as coarsening

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


def kernighan_lin(A, nb_cuts, fractions, verbose):
    cuts = []

    for f in fractions:
        if verbose >= 3:
            print(f"\t Calculating cuts for a fraction of: 1/{f}")
        for c in range(nb_cuts):
            cut = kernighan_lin_algorithm(A, 1 / f)
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
# Fiduccia-Mattheyses-Algorithm
#
# ----------------------------------------------------------------------------------------------------------------------

def fid_mat(xs, nb_cuts, ratio):
    cuts = []

    for c in range(nb_cuts):
        print("calculating cut: ", c)
        cut = fid_mat_algorithm(xs, ratio)
        cuts.append(cut)

    cuts = np.array(cuts)

    return cuts


def fid_mat_algorithm(xs, r):
    r = np.random.uniform(r, 0.5)
    nb_cells, _ = xs.shape
    A, B = initial_partition(xs, np.random.uniform(r, 0.5))

    cell_array = [np.argwhere(row == True).flatten() for row in xs]

    # p_max is the maximal degree and thus the maximal gain a vertex can have for a single move
    p_max = np.max(np.sum(xs, axis=1))

    # while not converged
    while True:
        A_copy = A.copy()
        B_copy = B.copy()
        not_locked = np.full([nb_cells], True)
        gain_bucket, gain_list = compute_initial_gains(A, B, cell_array, p_max)
        g_max = -np.inf
        g_acc = 0
        move_max = np.empty_like(A)
        move_acc = np.zeros_like(A)

        # iterate over all vertices and move them
        while sum(not_locked > 0):
            base_cell, g = choose_cell_greedy(A_copy, B_copy, gain_bucket, r, not_locked, p_max)

            if base_cell is None:
                break

            g_acc += g
            move_acc[base_cell] = True

            if g_acc > g_max:
                g_max = g_acc
                move_max[:] = move_acc[:]

            if A_copy[base_cell]:
                A_copy, B_copy, gain_bucket, gain_list, not_locked = \
                    move_and_update(base_cell, A_copy, B_copy, gain_bucket, gain_list, not_locked, cell_array)
            else:
                B_copy, A_copy, gain_bucket, gain_list, not_locked = \
                    move_and_update(base_cell, B_copy, A_copy, gain_bucket, gain_list, not_locked, cell_array)

        if g_max > 0:
            # moving nodes from initial partition that improve the cut
            np.logical_not(A, out=A, where=move_max)
            np.logical_not(A, out=B)
        else:
            break

    print("final ratio: ", sum(A) / nb_cells)

    return A


## g(i) = FromSingle(i) - ToEmpty(i)
# computed gains like in pseudocode but not sure if this is fastest? O(n^2)
def compute_initial_gains(A, B, cell_array, p_max):
    nb_cells = len(cell_array)
    gain_bucket = np.empty([2*p_max+1, ], object)
    gain_bucket[...] = [[] for _ in range(2*p_max+1)]
    gain_list = np.full([nb_cells], None)

    for cell_index in range(nb_cells):
        gain = 0
        for adj_cell in cell_array[cell_index]:
            if A[cell_index]:
                gain += compute_gain_for_net(A, B, adj_cell)
            elif B[cell_index]:
                gain += compute_gain_for_net(B, A, adj_cell)

        # add cell to the sorted bucket list
        gain_bucket[gain].append(cell_index)
        gain_list[cell_index] = gain

    return gain_bucket, gain_list


def compute_gain_for_net(F, T, other_index):
    Fn = F[other_index] + 1
    Tn = T[other_index]

    if Fn == 1:
        return +1
    elif Tn == 0:
        return -1
    else:
        return 0


def choose_cell_greedy(A, B, gain_bucket, r, not_locked, p_max):

    possible_partition = is_balanced(A, B, r, sum(not_locked))

    # choose cell that is not locked, does not harm the ratio and maximizes the gain
    for index in np.arange(p_max, -p_max-1, -1):
        for cell in gain_bucket[index]:
            partition = [A[cell], B[cell]]
            if not_locked[cell] & np.logical_and(partition, possible_partition).any():
                return cell, index

    return None, 0


def is_balanced(A, B, r, smax):
    sumA = sum(A)
    cardinalityA_1 = sumA - 1
    cardinalityA_2 = sumA + 1

    W = len(A)

    leftbound = r*W
    rightbound = W - r*W
    return [leftbound <= cardinalityA_1 <= rightbound, leftbound <= cardinalityA_2 <= rightbound]



def move_and_update(base_cell, F, T, gain_bucket, gain_list, not_locked, cell_array):
    # lock base cell
    not_locked[base_cell] = False
    # remove the base cell from the bucket list
    gain_bucket[gain_list[base_cell]].remove(base_cell)

    # switch block
    F[base_cell] = 0
    T[base_cell] = 1

    # increment or decrement gain of neighbouring cells
    for other_cell in cell_array[base_cell]:
        # check critical nets before move
        Tn = T[other_cell]
        # Fn = F[other_cell] + 1
        if not_locked[other_cell]:
            if Tn == 0:
                gain_bucket, gain_list = adjust_gain(gain_bucket, gain_list, other_cell, +2)
            elif Tn == 1:
                gain_bucket, gain_list = adjust_gain(gain_bucket, gain_list, other_cell, -2)

            # # chance net distribution to reflect the move
            # Tn += 1
            # Fn -= 1
            #
            # if Fn == 0:
            #     gain_bucket, gain_list = adjust_gain(gain_bucket, gain_list, other_cell, -1)
            # elif Fn == 1:
            #     gain_bucket, gain_list = adjust_gain(gain_bucket, gain_list, other_cell, +1)

    return F, T, gain_bucket, gain_list, not_locked


def adjust_gain(g_bucket, g_list, cell, value):
    g_bucket[g_list[cell]].remove(cell)
    g_list[cell] += value
    g_bucket[g_list[cell]].append(cell)

    return g_bucket, g_list


# ----------------------------------------------------------------------------------------------------------------------
# Coarsening approach
# ----------------------------------------------------------------------------------------------------------------------


def coarsening_cuts(A, nb_cuts, n_max):

    find_cuts = coarsening.FindCuts(A=A,
                                    merge_fn=coarsening.max_cut_merging,
                                    partition_fn=coarsening.compute_spectral_wcut)

    cuts = []

    for _ in np.arange(nb_cuts):
        cuts.append(find_cuts(N_max=n_max, verbose=False))

    cuts = np.array(cuts).astype(bool)
    return cuts


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
