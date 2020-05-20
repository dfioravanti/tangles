import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from sklearn.datasets import make_blobs

import networkx as nx


def load_eps_blobs(blob_sizes, blob_centers, blob_variances, eps, seed):

    xs, ys = make_blobs(n_samples=blob_sizes, centers=blob_centers, cluster_std=blob_variances, n_features=2, random_state=seed)
    A = radius_neighbors_graph(xs, eps).toarray()
    G = nx.from_numpy_matrix(A)

    return xs, ys, A, G
