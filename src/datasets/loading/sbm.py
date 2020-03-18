import numpy as np

import matplotlib.pyplot as plt
import networkx as nx


def load_sbm(block_size, nb_blocks, p_in, p_out):

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

    a = load_sbm(10, 4, .7, .3)
    print(a)
