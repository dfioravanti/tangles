from src.oriented_cuts import OrientedCut


def test_oriented_cuts():

    # Creation

    cuts, orientations = 0, True
    c = OrientedCut(cuts=cuts, orientations=orientations)
    assert c.get_cuts() == [cuts] and c.get_orientations() == [orientations]

    cuts = [0, 1, 2, 3, 4]
    orientations = [True, False, True, True, False]
    or_cuts = OrientedCut(cuts=cuts, orientations=orientations)
    for i, (c, o) in enumerate(or_cuts):
        assert c == cuts[i] and o == orientations[i]

    c1 = OrientedCut(cuts=0, orientations=True)
    c2 = OrientedCut(cuts=0, orientations=False)
    assert c1 == c1
    assert c1 != c2

    # Equality

    cuts = [0, 1, 2, 3, 4]
    o1 = [True, False, True, True, False]
    o2 = [True, False, True, True, True]
    c1 = OrientedCut(cuts=cuts, orientations=o1)
    c2 = OrientedCut(cuts=cuts, orientations=o2)
    assert c1 != c2

    # hashing

    h1 = hash(c1)
    h2 = hash(c2)
    assert h1 == h1
    assert h2 == h2
    assert h1 != h2

    # Addition

    cuts, cut = [0, 1, 2], 3
    orrs, orr = [True, False, True], True
    c1 = OrientedCut(cuts=cuts, orientations=orrs)
    c2 = OrientedCut(cuts=cuts, orientations=orrs)
    ref_cut = OrientedCut(cuts + [cut], orrs + [orr])
    c3, changed = c1.add(cut, orr)
    assert c1 == c2
    assert c1 != c3
    assert changed and c3 == ref_cut
    assert c3.size == c1.size + 1

    cuts, cut = [0, 1, 2], 1
    orrs, orr = [True, False, True], False

    c = OrientedCut(cuts=cuts, orientations=orrs)
    ref_cut = OrientedCut(cuts, orrs)
    c, changed = c.add(cut, orr)
    assert not changed and c == ref_cut

    cuts, cut = [0, 1, 2], 1
    orrs, orr = [True, False, True], True

    c = OrientedCut(cuts=cuts, orientations=orrs)
    ref_cut = OrientedCut(cuts, orrs)
    c, changed = c.add(cut, orr)
    assert not changed and c == ref_cut


if __name__ == '__main__':
    test_oriented_cuts()