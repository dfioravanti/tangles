import numpy as np
import pandas as pd

import networkx as nx


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


def load_SBM(block_sizes, p_in, p_out, seed):

    nb_nodes = np.sum(block_sizes)

    A = np.zeros((nb_nodes, nb_nodes), dtype=bool)
    ys = np.zeros(nb_nodes, dtype=int)
    G = nx.random_partition_graph(block_sizes, p_in, p_out, seed=seed)

    for node, ad in G.adjacency():
        A[node, list(ad.keys())] = True

    for cls, points in enumerate(G.graph["partition"]):
        ys[list(points)] = cls

    return A, ys, G


if __name__ == '__main__':

    a = load_SBM(10, 4, .7, .3)
    print(a)
