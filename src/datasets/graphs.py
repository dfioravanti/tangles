import numpy as np

import networkx as nx


def load_FLORENCE():

    G = nx.florentine_families_graph()
    A = nx.to_numpy_matrix(G, dtype=bool)

    return A, None, G


def load_ROC(nb_cliques, clique_size):

    nb_nodes = nb_cliques * clique_size

    A = np.zeros((nb_nodes, nb_nodes), dtype=bool)
    ys = np.zeros(nb_nodes, dtype=int)
    G = nx.ring_of_cliques(nb_cliques, clique_size)

    for node, ad in G.adjacency():
        A[node, list(ad.keys())] = True

    for cls in range(nb_cliques):
        ys[cls*clique_size:(cls+1)*clique_size] = cls

    return A, ys, G


def load_multilevel(nb_nodes, p_in, p_out):

    p = [[p_in[0], p_in[0], p_out, p_out],
         [p_in[0], p_in[1], p_out, p_out],
         [p_out, p_out, p_in[0], p_in[0]],
         [p_out, p_out, p_in[0], p_in[1]]]

    nb_nodes = [nb_nodes[0], nb_nodes[1], nb_nodes[0], nb_nodes[1]]
    nb_points = np.sum(nb_nodes)

    A = np.zeros((nb_points, nb_points), dtype=bool)
    ys = np.zeros(nb_points, dtype=int)
    G = nx.stochastic_block_model(sizes=nb_nodes, p=p)

    for node, ad in G.adjacency():
        A[node, list(ad.keys())] = True

    for cls, points in enumerate(G.graph["partition"]):
        ys[list(points)] = cls

    return A, ys, G


def load_LFR(nb_nodes, tau1, tau2, mu, min_community, average_degree, seed):

    A = np.zeros((nb_nodes, nb_nodes), dtype=bool)
    ys = np.zeros(nb_nodes, dtype=int)
    G = nx.LFR_benchmark_graph(nb_nodes, tau1, tau2, mu,
                                   min_community=min_community, average_degree=average_degree,
                                   seed=seed)

    for node, ad in G.adjacency():
        A[node, list(ad.keys())] = True

    partitions = {frozenset(G.nodes[v]['community']) for v in G}
    for cls, points in enumerate(partitions):
        ys[list(points)] = cls

    return A, ys, G


def load_RPG(block_sizes, p_in, p_out, seed):

    nb_nodes = np.sum(block_sizes)

    A = np.zeros((nb_nodes, nb_nodes), dtype=bool)
    ys = np.zeros(nb_nodes, dtype=int)
    G = nx.random_partition_graph(block_sizes, p_in, p_out, seed=seed)

    for node, ad in G.adjacency():
        A[node, list(ad.keys())] = True

    for cls, points in enumerate(G.graph["partition"]):
        ys[list(points)] = cls

    return A, ys, G


def load_SBM(block_size, nb_blocks, p_in, p_out):

    sizes = np.zeros(nb_blocks, dtype=int) + block_size
    p = np.zeros((nb_blocks, nb_blocks)) + p_out
    idx = np.diag_indices(nb_blocks)
    p[idx] = p_in

    nb_points = block_size * nb_blocks

    A = np.zeros((nb_points, nb_points), dtype=bool)
    ys = np.zeros(nb_points, dtype=int)
    G = nx.stochastic_block_model(sizes=sizes, p=p)

    for node, ad in G.adjacency():

        A[node, list(ad.keys())] = True

    for cls, points in enumerate(G.graph["partition"]):
        ys[list(points)] = cls

    return A, ys, G


if __name__ == '__main__':

    a = load_SBM(10, 4, .7, .3)
    print(a)
