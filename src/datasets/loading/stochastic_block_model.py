import numpy as np

from networkx.generators.community import stochastic_block_model


def load_sbm(block_size, nb_blocks, p_in, p_out):

    sizes = np.zeros(nb_blocks, dtype=int) + block_size
    p = np.zeros((nb_blocks, nb_blocks)) + p_out
    idx = np.diag_indices(nb_blocks)
    p[idx] = p_in

    nb_points = block_size * nb_blocks

    xs = np.zeros((nb_points, nb_points), dtype=bool)
    ys = np.zeros(nb_points, dtype=int)
    graph = stochastic_block_model(sizes=sizes, p=p)
    for node, ad in graph.adjacency():

        xs[node, list(ad.keys())] = True

    for cls, points in enumerate(graph.graph["partition"]):
        ys[list(points)] = cls

    return xs, ys


if __name__ == '__main__':

    load_sbm(10, 4, .7, .3)
