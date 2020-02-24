import unittest

from src.datasets.generation.questionnaire import make_centers
from src.misc.distances import min_center_distance


class TestQuestionnaire(unittest.TestCase):

    def test_make_centers(self):

        n_features, n_mindsets = 2, 2
        c = make_centers(n_features, n_mindsets)
        min_d = min_center_distance(c)
        self.assertGreater(min_d, (n_features // n_mindsets) - 2)

        n_features, n_mindsets = 3, 2
        c = make_centers(n_features, n_mindsets)
        min_d = min_center_distance(c)
        self.assertGreater(min_d, (n_features // n_mindsets) - 2)

        n_features, n_mindsets = 5, 4
        c = make_centers(n_features, n_mindsets)
        min_d = min_center_distance(c)
        self.assertGreater(min_d, (n_features // n_mindsets) - 2)

        n_features, n_mindsets = 10000, 977
        c = make_centers(n_features, n_mindsets)
        min_d = min_center_distance(c)
        self.assertGreater(min_d, (n_features // n_mindsets) - 2)

    def test_make_questionnaire(self):

        # TODO: Finish tests

        pass


if __name__ == '__main__':
    unittest.main()
