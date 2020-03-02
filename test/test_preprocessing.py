import numpy as np

from src.preprocessing import make_submodular


def test_make_submodular():
    xs = np.array([[0, 0, 0, 0],
                   [1, 1, 0, 0],
                   [0, 0, 1, 1],
                   [1, 1, 1, 1]], dtype=bool)

    cuts = np.array([xs[:, 0]])
    new_cuts = make_submodular(cuts)
    np.testing.assert_array_equal(cuts, new_cuts)

    cuts = np.array([xs[:, 0], xs[:, 1]])
    new_cuts = make_submodular(cuts)
    np.testing.assert_array_equal(new_cuts, [xs[:, 0]])

    cuts = np.array([[1, 0, 0, 0],
                     [0, 0, 0, 1]], dtype=bool)
    expected_new_cuts = np.array([[1, 0, 0, 0],
                                  [0, 0, 0, 1],
                                  [1, 0, 0, 1],
                                  [0, 1, 1, 1],
                                  [1, 1, 1, 0]], dtype=bool)
    new_cuts = make_submodular(cuts)
    np.testing.assert_array_equal(expected_new_cuts, new_cuts)

    cuts = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=bool)
    new_cuts = make_submodular(cuts)
    assert len(new_cuts) == 2**4 - 2


if __name__ == '__main__':
    test_make_submodular()