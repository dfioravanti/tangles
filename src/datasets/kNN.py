import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.datasets import make_blobs

import networkx as nx


def load_knn_blobs(blob_sizes, blob_centers, k, seed):

    xs, ys = make_blobs(n_samples=blob_sizes, centers=blob_centers, n_features=2, random_state=seed)
    A = kneighbors_graph(xs, k).toarray()
    G = nx.from_numpy_matrix(A)

    return xs, ys, A, G


# build knn from Gaussian
def load_KNN(mus, vars, size_blocks, nb_blocks, k):
    locations, labels = gaussian(mus, vars, np.array([size_blocks] * nb_blocks))

    W = kneighbors_graph(locations, k)

    A = np.array(W.todense())
    G = nx.from_numpy_matrix(A)
    return  labels, A, G


# sample locations
def gaussian(mus, vars, sizes):
    x = np.array([])
    y = np.array([])
    labels = np.array([])
    for i, size in enumerate(sizes):
        x_tmp, y_tmp = np.random.multivariate_normal(mus[i], np.diag(vars[i]), size).T
        x = np.concatenate([x, x_tmp])
        y = np.concatenate([y, y_tmp])
        labels = np.concatenate([labels, [i] * size])

    return np.array(list(zip(x, y))), np.array(labels)


# generate kNN graph
def kNN(positions, k):
    n = positions.shape[0]
    dists = euclidean_distances(positions, positions)
    idx = np.argsort(dists, axis=1)[:, 1:k + 1]
    W = np.zeros([n, n])
    for i in range(W.shape[0]):
        W[i, idx[i]] = 1
        W[idx[i], i] = 1

    return W
