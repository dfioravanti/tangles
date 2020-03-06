from copy import deepcopy

import pytest

from src.oriented_cuts import OrientedCut


def test_oriented_cuts():

    # Creation

    oriented_cut = {0: True}
    c = OrientedCut(oriented_cut)
    assert c.get_idx_cuts() == [0] and c.get_orientations() == [True]
    assert c.min_cuts == [0]

    cuts = [0, 1, 2, 3, 4]
    orientations = [True, False, True, True, False]
    oriented_cut = dict(zip(cuts, orientations))

    or_cuts = OrientedCut(oriented_cut)
    for i, (c, o) in enumerate(or_cuts.items()):
        assert c == cuts[i] and o == orientations[i]

    c1 = OrientedCut({0: True})
    c2 = OrientedCut({0: False})
    assert c1 == c1
    assert c1 != c2

    # Equality

    cuts = [0, 1, 2, 3, 4]
    o1 = [True, False, True, True, False]
    o2 = [True, False, True, True, True]
    oc1 = dict(zip(cuts, o1))
    oc2 = dict(zip(cuts, o2))

    c1 = OrientedCut(oc1)
    c2 = OrientedCut(oc2)
    assert c1 != c2

    # Get
    assert not c1.orientation_of(4)
    assert c2.orientation_of(4)
    assert c2.orientation_of(6) is None

    # hashing

    h1 = hash(c1)
    h2 = hash(c2)
    assert h1 == h1
    assert h2 == h2
    assert h1 != h2

    # Addition

    old_cuts, new_cut = [0, 1, 2], 3
    old_orrs, new_orr = [True, False, True], True
    ref_cuts, ref_orr = old_cuts + [new_cut], old_orrs + [new_orr]

    old_oriented_cs, new_oriented_c = dict(zip(old_cuts, old_orrs)), {new_cut: new_orr}
    ref_oriented_cs = dict(zip(ref_cuts, ref_orr))

    old_c = OrientedCut(old_oriented_cs)
    old_c2 = deepcopy(old_c)
    ref_cut = OrientedCut(ref_oriented_cs)

    new_c = old_c + OrientedCut(new_oriented_c)
    assert old_c == old_c2
    assert old_c != new_c
    assert new_c == ref_cut
    assert len(new_c) == len(old_c) + 1
    assert old_c.orientation_of(3) is None
    assert new_c.orientation_of(3)
    assert (OrientedCut({1: True}) + OrientedCut({1: False})) is None
    assert (OrientedCut({1: True}) + OrientedCut({1: True})) == OrientedCut({1: True})
    assert (OrientedCut({1: True, 2: True}) + OrientedCut({1: True, 2: False})) is None
    assert (OrientedCut({1: True, 2: False}) + OrientedCut({1: True, 2: False})) == OrientedCut({1: True, 2: False})
    assert (OrientedCut({1: False, 3: True}) + OrientedCut({1: True, 2: False})) is None
    assert (OrientedCut({1: True, 3: True}) + OrientedCut({1: True, 2: False})) == OrientedCut({1: True, 2: False, 3: True})


if __name__ == '__main__':
    test_oriented_cuts()
