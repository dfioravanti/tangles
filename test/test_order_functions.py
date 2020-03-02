import unittest

import numpy as np

from src.order_functions import implicit_order


class TestOrderFunctions(unittest.TestCase):

    def test_implicit_order(self):

        xs = np.array([[0, 0, 0],
                       [0, 1, 0]])

        # Test empty and all cuts
        cut = np.array([False, False])
        order = implicit_order(xs, cut)
        self.assertEqual(order, 0)
        cut = np.array([True, True])
        order = implicit_order(xs, cut)
        self.assertEqual(order, 0)

        # Test simple cuts
        cut = np.array([True, False])
        order = implicit_order(xs, cut)
        self.assertEqual(order, 1)
        cut = np.array([False, True])
        order = implicit_order(xs, cut)
        self.assertEqual(order, 1)

        # Test more than two elements cuts
        xs = np.array([[0, 0, 0, 0],
                       [1, 1, 0, 0],
                       [0, 0, 1, 1],
                       [1, 1, 1, 1]])
        cut = np.array([True, True, False, False])
        order = implicit_order(xs, cut)
        self.assertEqual(order, 3)

        # Test for silly n_samples
        order = implicit_order(xs, cut, n_samples=30)
        self.assertEqual(order, 3)


if __name__ == '__main__':
    unittest.main()
