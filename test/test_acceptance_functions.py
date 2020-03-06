import numpy as np

from src.acceptance_functions import triplet_size_big_enough
from src.oriented_cuts import OrientedCut


def test_size_too_small():

    all_cuts = np.array([[0, 1, 1, 0],
                         [0, 1, 1, 0],
                         [0, 0, 1, 1],
                         [0, 0, 1, 1]], dtype=bool)

    oriented_cuts = OrientedCut({0: True})
    assert triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=2)
    assert not triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=4)

    oriented_cuts = OrientedCut({2: True})
    assert triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=2)
    assert not triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=4)

    oriented_cuts = OrientedCut({0: True}).add_superset(OrientedCut({1: True}))
    assert triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=2)
    assert not triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=4)

    oriented_cuts = OrientedCut({0: True}).add_crossing(OrientedCut({1: False}))
    assert triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=0)
    assert not triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=1)

    oriented_cuts = [OrientedCut({0: True}).add_crossing(OrientedCut({1: True})), OrientedCut({2: False})]
    assert triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=1)
    assert not triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=2)

    oriented_cuts = [OrientedCut({0: True}).add_superset(OrientedCut({1: True})),
                     OrientedCut({2: False}), OrientedCut({3: True})]
    assert triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=0)
    assert not triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=1)

    oriented_cuts = [OrientedCut({0: True}).add_superset(OrientedCut({1: True})),
                     OrientedCut({2: True}).add_superset(OrientedCut({3: True}))]

    assert triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=0)
    assert triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=1)
    assert not triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=2)

    oriented_cuts = [OrientedCut({0: True}).add_superset(OrientedCut({1: True})),
                     OrientedCut({2: False}).add_crossing(OrientedCut({3: True}))]

    assert triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=0)
    assert not triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=1)

    all_cuts = np.array([[1, 1, 1, 1, 1],
                         [1, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 1]], dtype=bool)

    oriented_cuts = [OrientedCut({1: True}).add_superset(OrientedCut({0: True})),
                     OrientedCut({2: True}).add_superset(OrientedCut({0: True})),
                     OrientedCut({3: True}).add_superset(OrientedCut({0: True})),
                     OrientedCut({4: True}).add_superset(OrientedCut({0: True})),
                     OrientedCut({5: True}).add_superset(OrientedCut({0: True}))]

    assert triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=0)
    assert not triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=1)

    oriented_cuts = [OrientedCut({1: False}).add_superset(OrientedCut({0: True})),
                     OrientedCut({2: True}).add_superset(OrientedCut({0: True})),
                     OrientedCut({3: True}).add_superset(OrientedCut({0: True})),
                     OrientedCut({4: True}).add_superset(OrientedCut({0: True})),
                     OrientedCut({5: True}).add_superset(OrientedCut({0: True}))]
    assert triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=0)
    assert not triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=1)

    oriented_cuts = [OrientedCut({1: False}).add_superset(OrientedCut({0: True})),
                     OrientedCut({2: False}).add_superset(OrientedCut({0: True})),
                     OrientedCut({3: False}).add_superset(OrientedCut({0: True})),
                     OrientedCut({4: False}).add_superset(OrientedCut({0: True})),
                     OrientedCut({5: False}).add_superset(OrientedCut({0: True}))]
    assert triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=0)
    assert triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=1)
    assert triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=2)
    assert triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=3)
    assert not triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=4)
    assert not triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=5)

    all_cuts = np.array([[1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 1],
                         [0, 0, 1, 1, 1],
                         [0, 0, 0, 1, 1],
                         [0, 0, 0, 0, 1]], dtype=bool)

    oriented_cuts = [OrientedCut({2: True}).add_superset(OrientedCut({1: True})).add_superset(OrientedCut({0: True})),
                     OrientedCut({4: True}).add_superset(OrientedCut({3: True}))]
    assert triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=0)
    assert triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=1)
    assert not triplet_size_big_enough(all_cuts=all_cuts, oriented_cuts=oriented_cuts, min_size=2)


if __name__ == '__main__':
    test_size_too_small()
