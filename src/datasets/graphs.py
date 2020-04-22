import numpy as np
import pandas as pd

import networkx as nx


def load_FLORENCE():

    G = nx.florentine_families_graph()
    A = nx.to_numpy_matrix(G, dtype=bool)

    return A, None, G


def load_POLI_BOOKS(path_nodes, path_edges):

    nodes = pd.read_csv(path_nodes)
    ys = nodes['political_ideology'].astype('category').cat.codes.values

    file_edges = open(path_edges, 'rb')
    file_edges.readline()
    G = nx.read_edgelist(file_edges, delimiter=',', nodetype=int, data=(('weight', int),))
    A = nx.adjacency_matrix(G).todense()
    A = np.asarray(A)

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
