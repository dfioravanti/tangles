import numpy as np

from src.acceptance_functions import triplet_size_big_enough
from src.oriented_cuts import OrientedCut


def test_size_too_small():
    xs = np.array([[0, 0, 0, 0],
                   [1, 1, 0, 0],
                   [0, 0, 1, 1],
                   [1, 1, 1, 1]], dtype=bool)
    all_cuts = xs.T

    oriented_cuts = OrientedCut({0: True})
    assert triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=2)

    oriented_cuts = OrientedCut({0: True})
    assert not triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=4)

    oriented_cuts = OrientedCut({2: True})
    assert triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=2)

    oriented_cuts = OrientedCut({2: True})
    assert not triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=4)

    oriented_cuts = [OrientedCut({0: True}), OrientedCut({1: True})]
    assert triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=2)

    oriented_cuts = [OrientedCut({0: True}), OrientedCut({1: True})]
    assert not triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=4)

    oriented_cuts = [OrientedCut({0: True}), OrientedCut({1: False})]
    assert triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=0)

    oriented_cuts = [OrientedCut({0: True}), OrientedCut({1: False})]
    assert not triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=1)

    oriented_cuts = [OrientedCut({0: True, 1: True}), OrientedCut({2: False})]
    assert triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=1)

    oriented_cuts = [OrientedCut({0: True, 1: True}), OrientedCut({2: False})]
    assert not triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=2)

    oriented_cuts = [OrientedCut({0: True, 1: True}), OrientedCut({2: False}), OrientedCut({3: True})]
    assert not triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=1)

    oriented_cuts = [OrientedCut({0: True, 1: True}), OrientedCut({2: True}), OrientedCut({3: True})]
    assert triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=1)

    oriented_cuts = OrientedCut({0: True, 1: True, 2: False, 3: True})
    assert not triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=1)

    oriented_cuts = OrientedCut({0: True, 1: True, 2: True, 3: True})
    assert triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=1)

    xs = np.array([[1, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, 0, 1, 0],
                   [1, 1, 1, 1, 1]], dtype=bool)
    all_cuts = xs.T

    oriented_cuts = [OrientedCut({0: True, 1: True, 2: True, 3: True}), OrientedCut({4: True})]
    assert triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=1)

    oriented_cuts = [OrientedCut({0: True, 1: True, 2: True, 3: True}), OrientedCut({4: False})]
    assert not triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=1)


if __name__ == '__main__':
    test_size_too_small()
