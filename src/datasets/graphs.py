import numpy as np

import matplotlib.pyplot as plt
import networkx as nx


def load_ROC(nb_cliques, clique_size):

    nb_nodes = nb_cliques * clique_size

    A = np.zeros((nb_nodes, nb_nodes), dtype=bool)
    ys = np.zeros(nb_nodes, dtype=int)
    graph = nx.ring_of_cliques(nb_cliques, clique_size)

    for node, ad in graph.adjacency():
        A[node, list(ad.keys())] = True

    for cls in range(nb_cliques):
        ys[cls*clique_size:(cls+1)*clique_size] = cls

    pos = nx.spring_layout(graph, k=2)
    nx.draw(graph, pos, arrows=False, node_color=ys, vmin=0, vmax=max(ys), cmap='Set1')
    plt.savefig('graph.svg')

    return A, ys


def load_LFR(nb_nodes, tau1, tau2, mu, min_community, average_degree, seed):

    A = np.zeros((nb_nodes, nb_nodes), dtype=bool)
    ys = np.zeros(nb_nodes, dtype=int)
    graph = nx.LFR_benchmark_graph(nb_nodes, tau1, tau2, mu,
                                   min_community=min_community, average_degree=average_degree,
                                   seed=seed)

    for node, ad in graph.adjacency():
        A[node, list(ad.keys())] = True

    partitions = {frozenset(graph.nodes[v]['community']) for v in graph}
    for cls, points in enumerate(partitions):
        ys[list(points)] = cls

    pos = nx.spring_layout(graph, k=2)
    nx.draw(graph, pos, arrows=False, node_color=ys, vmin=0, vmax=max(ys), cmap='Set1')
    plt.savefig('graph.svg')

    return A, ys


def load_RPG(block_size, nb_blocks, p_in, p_out):

    sizes = np.zeros(nb_blocks, dtype=int) + block_size
    nb_nodes = block_size * nb_blocks

    A = np.zeros((nb_nodes, nb_nodes), dtype=bool)
    ys = np.zeros(nb_nodes, dtype=int)
    graph = nx.random_partition_graph(sizes, p_in, p_out)

    for node, ad in graph.adjacency():
        A[node, list(ad.keys())] = True

    for cls, points in enumerate(graph.graph["partition"]):
        ys[list(points)] = cls

    pos = nx.spring_layout(graph, k=2)
    nx.draw(graph, pos, arrows=False, node_color=ys, vmin=0, vmax=max(ys), cmap='Set1')
    plt.savefig('graph.svg')

    return A, ys


def load_SBM(block_size, nb_blocks, p_in, p_out):

    sizes = np.zeros(nb_blocks, dtype=int) + block_size
    p = np.zeros((nb_blocks, nb_blocks)) + p_out
    idx = np.diag_indices(nb_blocks)
    p[idx] = p_in

    nb_points = block_size * nb_blocks

    A = np.zeros((nb_points, nb_points), dtype=bool)
    ys = np.zeros(nb_points, dtype=int)
    graph = nx.stochastic_block_model(sizes=sizes, p=p)

    for node, ad in graph.adjacency():

        A[node, list(ad.keys())] = True

    for cls, points in enumerate(graph.graph["partition"]):
        ys[list(points)] = cls

    pos = nx.spring_layout(graph, k=2)
    nx.draw(graph, pos, arrows=False, node_color=ys, vmin=0, vmax=max(ys),  cmap='Set1')
    plt.savefig('graph.svg')

    return A, ys


if __name__ == '__main__':

    a = load_SBM(10, 4, .7, .3)
    print(a)
