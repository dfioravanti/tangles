import numpy as np
from scipy.linalg import hadamard
from src.misc import count_difference


def make_centers(n_features=20, n_mindsets=2):

    """
    Generates n_mindsets centers that are have more or less half of their features different

    TODO: Check out if we can use some other type of code that have a guaranteed Hemming distance.

    Parameters
    ----------
    n_features
    n_mindsets

    Returns
    -------

    """

    next_power_two = np.int(np.power(2, np.ceil(np.log2(n_features))))
    h_matrix = hadamard(next_power_two)
    h_matrix = (h_matrix == 1)

    idxs = np.arange(next_power_two)
    idxs = np.random.choice(idxs, n_mindsets, replace=False)

    return h_matrix[idxs, :n_features]


def make_synthetic_questionnaire(n_samples=100, n_features=20, n_mindsets=2, tolerance=0.2,
                                 centers=False):
    """

    This function simulates a synthetic questionnaire.

    The center of the mindsets will always disagree among each other in n_features / n_mindsets features.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of samples.
    n_features : int, optional (default=20)
        The number of features.
    n_mindsets : int, optional (default=2)
        The number of classes, we call them "mindsets", that we should generate.
    tolerance: float, optional (default=0.2)
        The percentage of deviation, computed respected to the 0-1 loss, that we allow
         inside a mindset. Must be in [0,1].
    centers : bool, optional (default=False)
        If True the function will return an additional array that contains the
        coordinates of the center of the mindset.

    Returns
    -------

    X : array of shape [n_samples, n_features]
        The generated samples.
    y : array of shape [n_samples]
        The integer labels for mindset membership of each sample.
    c : array of shape [n_mindsets, n_features]
        The coordinates of the centers of the mindsets

    """

    # TODO: We need some check on the number of mindsets vs number of samples and features.
    #       think how to implement that.

    if not 0 < tolerance < 1:
        raise ValueError("tolerance must be in [0,1]")

    c = make_centers(n_features, n_mindsets)
    max_n_errors = np.floor(n_features * tolerance).astype(int)

    idx_errors = np.random.choice(n_features, size=(n_samples, max_n_errors))
    errors = np.zeros((n_samples, n_features))
    errors[idx_errors] = 1
    # TODO: FInish loop stuff
    x = c + errors

    return c


if __name__ == '__main__':
    make_synthetic_questionnaire()