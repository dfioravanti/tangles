import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import kneighbors_graph

# build knn from Gaussian
def gaussiankNN(mus, vars, sizes, k):
    locations, labels = gaussian(mus, vars, sizes)

    W = kneighbors_graph(locations, k)

    A = np.array(W)

    return A


# sample locations
def gaussian(mus, vars, sizes):
    x = np.array([])
    y = np.array([])
    labels = np.array([])
    for i, size in enumerate(sizes):
        x_tmp, y_tmp = np.random.multivariate_normal(mus[i], vars[i], size).T
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
