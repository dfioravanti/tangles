import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from sklearn.datasets import make_blobs

import networkx as nx


def load_knn_gauss_blobs(blob_sizes, blob_centers, blob_variances, k, seed):

    xs, ys = make_blobs(n_samples=blob_sizes, centers=blob_centers, cluster_std=blob_variances, n_features=2, random_state=seed, shuffle=False)
    A = kneighbors_graph(xs, k, mode="distance").toarray()

    sigma = np.median(A[A > 0])

    A[A > 0] = np.exp(- A[A > 0] / (2*sigma**2))

    G = nx.from_numpy_matrix(A)
    return xs, ys, A, G
