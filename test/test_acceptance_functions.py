import numpy as np

from src.acceptance_functions import size_too_small
from src.oriented_cuts import OrientedCut


def test_size_too_small():

    xs = np.array([[0, 0, 0, 0],
                   [1, 1, 0, 0],
                   [0, 0, 1, 1],
                   [1, 1, 1, 1]], dtype=bool)
    all_cuts = xs.T

    oriented_cuts = OrientedCut(cuts=0, orientations=True)
    assert size_too_small(all_cuts, min_size=4, oriented_cuts=oriented_cuts)
    oriented_cuts = OrientedCut(cuts=0, orientations=True)
    assert not size_too_small(all_cuts, min_size=2, oriented_cuts=oriented_cuts)

    oriented_cuts = OrientedCut(cuts=2, orientations=False)
    assert size_too_small(all_cuts, min_size=4, oriented_cuts=oriented_cuts)
    oriented_cuts = OrientedCut(cuts=2, orientations=False)
    assert not size_too_small(all_cuts, min_size=2, oriented_cuts=oriented_cuts)

    oriented_cuts = OrientedCut(cuts=[0, 1], orientations=[True, True])
    assert size_too_small(all_cuts, min_size=4, oriented_cuts=oriented_cuts)
    oriented_cuts = OrientedCut(cuts=[0, 1], orientations=[True, True])
    assert not size_too_small(all_cuts, min_size=2, oriented_cuts=oriented_cuts)

    oriented_cuts = OrientedCut(cuts=[0, 0], orientations=[True, False])
    assert not size_too_small(all_cuts, min_size=0, oriented_cuts=oriented_cuts)
    oriented_cuts = OrientedCut(cuts=[0, 0], orientations=[True, False])
    assert size_too_small(all_cuts, min_size=1, oriented_cuts=oriented_cuts)

    c1 = OrientedCut(cuts=[0], orientations=[True, False])


if __name__ == '__main__':
    test_size_too_small()