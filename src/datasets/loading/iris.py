import numpy as np

from sklearn.datasets import load_iris
from sklearn.cluster import k_means
from sklearn.preprocessing import OneHotEncoder


def get_binarized_iris(k=3):

    """
    Build a binarized version of the iris dataset.
    Every feature is clustered with a k-means++ algorithm and then one-hot
    encoded to make it binary.

    Parameters
    ----------

    k : int, optional (default=3)
        Number of cluster to be used in the k-means algorithm

    Returns
    -------

    binarized_xs: array of shape [150, 4 * k]
        The binarized version of the iris dataset
    ys: array of shape [150]
        The labels

    """

    xs, ys = load_iris(return_X_y=True)
    binarized_xs = []

    for feature in xs.T:
        _, clustered_feature, _, = k_means(feature.reshape(-1, 1), k)
        enc = OneHotEncoder()
        binarized_feature = enc.fit_transform(clustered_feature.reshape(-1, 1)).toarray()
        binarized_feature = binarized_feature.astype(bool)
        binarized_xs.append(binarized_feature)

    binarized_xs = np.concatenate(binarized_xs, axis=1)

    return binarized_xs, ys


if __name__ == '__main__':
    get_binarized_iris()
