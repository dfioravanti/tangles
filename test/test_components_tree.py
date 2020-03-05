import numpy as np

from src.components_tree import Node


def test_add():
    cuts = np.array([[1, 1, 1, 1],
                     [0, 1, 1, 1],
                     [0, 0, 1, 1],
                     [0, 0, 0, 1]], dtype=bool)

    tree = Node(cuts[0], 0)
    for i_cut in np.arange(1, len(cuts)):
        tree, flip_cut = tree.insert(cuts[i_cut], i_cut)

    assert tree.idx == {0, 1, 2, 3}
    assert not tree.leaf
    assert tree.subsets.idx == {1, 2, 3}
    assert not tree.subsets.leaf
    assert tree.subsets.subsets.idx == {2, 3}
    assert not tree.subsets.subsets.leaf
    assert tree.subsets.subsets.subsets.idx == {3}
    assert tree.subsets.subsets.subsets.leaf
    assert tree.subsets.subsets.subsets.subsets is None

    cuts = np.array([[0, 1, 1, 0],
                     [1, 0, 0, 1],
                     [0, 0, 1, 1],
                     [0, 0, 0, 1]], dtype=bool)

    ref_cuts = np.array([[0, 1, 1, 0],
                         [0, 1, 1, 0],
                         [0, 0, 1, 1],
                         [1, 1, 1, 0]], dtype=bool)

    tree = Node(cuts[0], 0)
    for i_cut in np.arange(1, len(cuts)):
        tree, flip_cut = tree.insert(cuts[i_cut], i_cut)
        if flip_cut:
            cuts[i_cut] = ~cuts[i_cut]

    assert tree.idx == {0, 1, 2}
    assert not tree.leaf
    assert tree.subsets.idx == {1, 2}
    assert not tree.subsets.leaf
    assert tree.subsets.subsets.idx == {1, 2}
    assert tree.subsets.subsets.leaf
    assert tree.incomps.idx == {2, 3}
    assert tree.incomps.leaf
    assert np.all(cuts == ref_cuts)

    cuts = np.array([[0, 1, 1, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 1, 1, 1, 0],
                     [0, 0, 1, 0, 1, 0],
                     [1, 1, 1, 0, 0, 0],
                     [1, 0, 1, 0, 0, 0]], dtype=bool)

    tree = Node(cuts[0], 0)
    for i_cut in np.arange(1, len(cuts)):
        tree, flip_cut = tree.insert(cuts[i_cut], i_cut)
        if flip_cut:
            cuts[i_cut] = ~cuts[i_cut]

    assert tree.idx == {0, 1, 2}
    assert tree.subsets.idx == {1, 2}
    assert tree.subsets.incomps.idx == {0, 2}
    assert tree.subsets.subsets.idx == {2}
    assert tree.incomps.idx == {2, 3, 4}
    assert tree.incomps.subsets.idx == {2, 4}

if __name__ == '__main__':
    test_add()
