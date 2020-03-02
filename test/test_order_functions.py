import numpy as np

from src.order_functions import implicit_order


def test_implicit_order():

    xs = np.array([[0, 0, 0],
                   [0, 1, 0]])

    # Test empty and all cuts
    cut = np.array([False, False])
    order = implicit_order(xs, cut)
    assert order == 0

    cut = np.array([True, True])
    order = implicit_order(xs, cut)
    assert order == 0

    # Test simple cuts
    cut = np.array([True, False])
    order = implicit_order(xs, cut)
    assert order == 1

    cut = np.array([False, True])
    order = implicit_order(xs, cut)
    assert order == 1

    # Test more than two elements cuts
    xs = np.array([[0, 0, 0, 0],
                   [1, 1, 0, 0],
                   [0, 0, 1, 1],
                   [1, 1, 1, 1]], dtype=bool)
    cut = np.array(xs[:, 0])
    order = implicit_order(xs, cut)
    assert order == 3

    # Test for silly n_samples
    order = implicit_order(xs, cut, n_samples=30)
    assert order == 3


