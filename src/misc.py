from sklearn.metrics import zero_one_loss


def count_difference(x, y):
    """
    Returns the number of mismatches between two binary vectors.

    Parameters
    ----------
    x
    y

    Returns
    -------

    """
    return zero_one_loss(x, y, normalize=False)