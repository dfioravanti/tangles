import networkx as nx
import numpy as np
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph


def calculate_knn_graph(data, k):
    data.A = kneighbors_graph(data.xs, k).toarray()
    data.G = nx.from_numpy_matrix(data.A)

    return data


def calculate_radius_graph(data, eps):

    data.A = radius_neighbors_graph(data.xs, eps).toarray()
    data.G = nx.from_numpy_matrix(data.A)

    return data


def calculate_weighted_knn_graph(data, k):

    data.A = kneighbors_graph(data.xs, k).toarray()

    sigma = np.median(data.A[data.A > 0])

    data.A[data.A > 0] = np.exp(- data.A[data.A > 0] / (2 * sigma ** 2))

    data.G = nx.from_numpy_matrix(data.A)

    return data