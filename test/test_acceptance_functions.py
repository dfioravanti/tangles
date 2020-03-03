import numpy as np

from src.acceptance_functions import triplet_size_big_enough
from src.oriented_cuts import OrientedCut


def test_size_too_small():
    xs = np.array([[0, 0, 0, 0],
                   [1, 1, 0, 0],
                   [0, 0, 1, 1],
                   [1, 1, 1, 1]], dtype=bool)
    all_cuts = xs.T

    old_oriented_cuts = OrientedCut()
    new_oriented_cut = OrientedCut(idx_cuts=0, orientations=True)
    assert triplet_size_big_enough(all_cuts=all_cuts, old_oriented_cuts=old_oriented_cuts,
                                   new_oriented_cut=new_oriented_cut, min_size=2)

    old_oriented_cuts = OrientedCut()
    new_oriented_cut = OrientedCut(idx_cuts=0, orientations=True)
    assert not triplet_size_big_enough(all_cuts=all_cuts, old_oriented_cuts=old_oriented_cuts,
                                       new_oriented_cut=new_oriented_cut, min_size=4)

    old_oriented_cuts = OrientedCut()
    new_oriented_cut = OrientedCut(idx_cuts=2, orientations=True)
    assert triplet_size_big_enough(all_cuts=all_cuts, old_oriented_cuts=old_oriented_cuts,
                                   new_oriented_cut=new_oriented_cut, min_size=2)

    old_oriented_cuts = OrientedCut()
    new_oriented_cut = OrientedCut(idx_cuts=2, orientations=True)
    assert not triplet_size_big_enough(all_cuts=all_cuts, old_oriented_cuts=old_oriented_cuts,
                                       new_oriented_cut=new_oriented_cut, min_size=4)

    old_oriented_cuts = OrientedCut(idx_cuts=0, orientations=True)
    new_oriented_cut = OrientedCut(idx_cuts=1, orientations=True)
    assert triplet_size_big_enough(all_cuts=all_cuts, old_oriented_cuts=old_oriented_cuts,
                                   new_oriented_cut=new_oriented_cut, min_size=2)

    old_oriented_cuts = OrientedCut(idx_cuts=0, orientations=True)
    new_oriented_cut = OrientedCut(idx_cuts=1, orientations=True)
    assert not triplet_size_big_enough(all_cuts=all_cuts, old_oriented_cuts=old_oriented_cuts,
                                       new_oriented_cut=new_oriented_cut, min_size=4)

    old_oriented_cuts = OrientedCut(idx_cuts=0, orientations=True)
    new_oriented_cut = OrientedCut(idx_cuts=1, orientations=False)
    assert triplet_size_big_enough(all_cuts=all_cuts, old_oriented_cuts=old_oriented_cuts,
                                   new_oriented_cut=new_oriented_cut, min_size=0)

    old_oriented_cuts = OrientedCut(idx_cuts=0, orientations=True)
    new_oriented_cut = OrientedCut(idx_cuts=1, orientations=False)
    assert not triplet_size_big_enough(all_cuts=all_cuts, old_oriented_cuts=old_oriented_cuts,
                                       new_oriented_cut=new_oriented_cut, min_size=1)

    old_oriented_cuts = OrientedCut(idx_cuts=[0, 1], orientations=[True, True])
    new_oriented_cut = OrientedCut(idx_cuts=2, orientations=False)
    assert triplet_size_big_enough(all_cuts=all_cuts, old_oriented_cuts=old_oriented_cuts,
                                   new_oriented_cut=new_oriented_cut, min_size=1)

    old_oriented_cuts = OrientedCut(idx_cuts=[0, 1], orientations=[True, True])
    new_oriented_cut = OrientedCut(idx_cuts=2, orientations=False)
    assert not triplet_size_big_enough(all_cuts=all_cuts, old_oriented_cuts=old_oriented_cuts,
                                       new_oriented_cut=new_oriented_cut, min_size=2)

    old_oriented_cuts = OrientedCut(idx_cuts=[0, 1, 2], orientations=[True, True, True])
    new_oriented_cut = OrientedCut(idx_cuts=3, orientations=True)
    assert triplet_size_big_enough(all_cuts=all_cuts, old_oriented_cuts=old_oriented_cuts,
                                   new_oriented_cut=new_oriented_cut, min_size=1)

    old_oriented_cuts = OrientedCut(idx_cuts=[0, 1, 2], orientations=[True, True, False])
    new_oriented_cut = OrientedCut(idx_cuts=3, orientations=True)
    assert not triplet_size_big_enough(all_cuts=all_cuts, old_oriented_cuts=old_oriented_cuts,
                                       new_oriented_cut=new_oriented_cut, min_size=1)

    xs = np.array([[1, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, 0, 1, 0],
                   [1, 1, 1, 1, 1]], dtype=bool)
    all_cuts = xs.T

    old_oriented_cuts = OrientedCut(idx_cuts=[0, 1, 2, 3], orientations=[True, True, True, True])
    new_oriented_cut = OrientedCut(idx_cuts=4, orientations=True)
    assert triplet_size_big_enough(all_cuts=all_cuts, old_oriented_cuts=old_oriented_cuts,
                                   new_oriented_cut=new_oriented_cut, min_size=1)

    old_oriented_cuts = OrientedCut(idx_cuts=[0, 1, 2, 3], orientations=[True, True, True, True])
    new_oriented_cut = OrientedCut(idx_cuts=4, orientations=False)
    assert not triplet_size_big_enough(all_cuts=all_cuts, old_oriented_cuts=old_oriented_cuts,
                                       new_oriented_cut=new_oriented_cut, min_size=1)

if __name__ == '__main__':
    test_size_too_small()
