import sys
import bitarray as ba
import numpy as np
from src.execution import tangle_computation
from src.tangles import Tangle, core_algorithm
from src.tree_tangles import TangleTree, TangleNode, ContractedTangleTree
from src.my_types import Cuts

sys.path.append("../src")


# class Test_tangle_computation(object):
#
#     def test_one_point_case(self):
#         values = np.zeros((1, 1), dtype=bool)
#         costs = np.zeros(1, dtype=float)
#
#         values[0, 0] = True
#         costs[0] = 1
#         cuts = Cuts(values=values, costs=costs)
#
#         agreement = 1
#
#         tangle_tree = tangle_computation(cuts=cuts, agreement=agreement, verbose=0)
#         assert tangle_tree.__class__ == TangleTree
#
#         assert tangle_tree.root.left_child is not None
#         assert tangle_tree.root.right_child is None
#
#         node = tangle_tree.root.left_child
#         assert node.last_cut_added_id == 0
#         assert node.last_cut_added_orientation == values[0]
#         assert node.tangle.specification == {0: True}
#         assert node.tangle.cuts == [ba.bitarray(values.tolist())]
#         assert node.tangle.core == [ba.bitarray(values.tolist())]
#
#         values[0, 0] = False
#         costs[0] = 1
#         cuts = Cuts(values=values, costs=costs)
#
#         agreement = 1
#
#         tangle_tree = tangle_computation(cuts=cuts, agreement=agreement, verbose=0)
#         assert tangle_tree.__class__ == TangleTree
#
#         assert tangle_tree.root.left_child is None
#         assert tangle_tree.root.right_child is not None
#
#         node = tangle_tree.root.right_child
#         assert node.last_cut_added_id == 0
#         assert node.last_cut_added_orientation == values[0]
#         assert node.tangle.specification == {0: False}
#         assert node.tangle.cuts == [ba.bitarray(values.tolist())]
#         assert node.tangle.core == [ba.bitarray(values.tolist())]
#
#         values = np.zeros((2, 1), dtype=bool)
#         costs = np.zeros(2, dtype=float)
#
#         values[:, 0] = [True, False]
#         costs[:] = [1, 1]
#         cuts = Cuts(values=values, costs=costs)
#
#         agreement = 1
#
#         tangle_tree = tangle_computation(cuts=cuts, agreement=agreement, verbose=0)
#         assert tangle_tree.__class__ == TangleTree
#
#         assert tangle_tree.root.left_child is not None
#         assert tangle_tree.root.right_child is None
#
#         l_node = tangle_tree.root.left_child
#         assert l_node.last_cut_added_id == 0
#         assert l_node.last_cut_added_orientation == values[0, :]
#         assert l_node.tangle.specification == {0: True}
#         assert l_node.tangle.cuts == [ba.bitarray(values[0].tolist())]
#         assert l_node.tangle.core == [ba.bitarray(values[0].tolist())]
#
#         assert l_node.left_child is None
#         assert l_node.right_child is not None
#
#         lr_node = l_node.right_child
#         assert lr_node.last_cut_added_id == 1
#         assert lr_node.last_cut_added_orientation == values[1, :]
#         assert lr_node.tangle.specification == {0: True, 1: False}
#
#         assert lr_node.tangle.cuts == [ba.bitarray([1]), ba.bitarray([1])]
#         assert lr_node.tangle.core == [ba.bitarray([1])]
#
#     def test_tree_points_case_permutation_one(self):
#         values = np.array([[1, 0, 0],
#                            [0, 1, 0],
#                            [0, 0, 1]]).astype(bool)
#         costs = np.array([1, 1, 1])
#         cuts = Cuts(values=values, costs=costs)
#
#         agreement = 1
#
#         tangle_tree = tangle_computation(cuts=cuts, agreement=agreement, verbose=0)
#         assert tangle_tree.__class__ == TangleTree
#
#         assert len(tangle_tree.maximals) == 3
#
#         node_lrr = tangle_tree.maximals[0]
#         node_rlr = tangle_tree.maximals[1]
#         node_rrl = tangle_tree.maximals[2]
#
#         assert node_lrr == tangle_tree.root.left_child.right_child.right_child
#         assert node_rlr == tangle_tree.root.right_child.left_child.right_child
#         assert node_rrl == tangle_tree.root.right_child.right_child.left_child
#
#         assert node_lrr.last_cut_added_id == 2
#         assert node_lrr.last_cut_added_orientation is False
#         assert node_lrr.tangle.specification == {0: True, 1: False, 2: False}
#
#         assert node_lrr.tangle.cuts == [ba.bitarray([1, 0, 0]), ba.bitarray([1, 0, 1]), ba.bitarray([1, 1, 0])]
#         assert node_lrr.tangle.core == [ba.bitarray([1, 0, 0])]
#
#         assert node_rlr.last_cut_added_id == 2
#         assert node_rlr.last_cut_added_orientation is False
#         assert node_rlr.tangle.specification == {0: False, 1: True, 2: False}
#
#         assert node_rlr.tangle.cuts == [ba.bitarray([0, 1, 1]), ba.bitarray([0, 1, 0]), ba.bitarray([1, 1, 0])]
#         assert node_rlr.tangle.core == [ba.bitarray([0, 1, 0])]
#
#         assert node_rrl.last_cut_added_id == 2
#         assert node_rrl.last_cut_added_orientation is True
#         assert node_rrl.tangle.specification == {0: False, 1: False, 2: True}
#
#         assert node_rrl.tangle.cuts == [ba.bitarray([0, 1, 1]), ba.bitarray([1, 0, 1]), ba.bitarray([0, 0, 1])]
#         assert node_rrl.tangle.core == [ba.bitarray([0, 0, 1])]
#
#     def test_tree_points_case_permutation_two(self):
#         values = np.array([[0, 1, 0],
#                            [1, 0, 0],
#                            [0, 0, 1]]).astype(bool)
#         costs = np.array([1, 1, 1])
#         cuts = Cuts(values=values, costs=costs)
#
#         agreement = 1
#
#         tangle_tree = tangle_computation(cuts=cuts, agreement=agreement, verbose=0)
#         assert tangle_tree.__class__ == TangleTree
#
#         assert len(tangle_tree.maximals) == 3
#
#         node_lrr = tangle_tree.maximals[0]
#         node_rlr = tangle_tree.maximals[1]
#         node_rrl = tangle_tree.maximals[2]
#
#         assert node_lrr == tangle_tree.root.left_child.right_child.right_child
#         assert node_rlr == tangle_tree.root.right_child.left_child.right_child
#         assert node_rrl == tangle_tree.root.right_child.right_child.left_child
#
#         assert node_lrr.last_cut_added_id == 2
#         assert node_lrr.last_cut_added_orientation is False
#         assert node_lrr.tangle.specification == {0: True, 1: False, 2: False}
#
#         assert node_lrr.tangle.cuts == [ba.bitarray([0, 1, 0]), ba.bitarray([0, 1, 1]), ba.bitarray([1, 1, 0])]
#         assert node_lrr.tangle.core == [ba.bitarray([0, 1, 0])]
#
#         assert node_rlr.last_cut_added_id == 2
#         assert node_rlr.last_cut_added_orientation is False
#         assert node_rlr.tangle.specification == {0: False, 1: True, 2: False}
#
#         assert node_rlr.tangle.cuts == [ba.bitarray([1, 0, 1]), ba.bitarray([1, 0, 0]), ba.bitarray([1, 1, 0])]
#         assert node_rlr.tangle.core == [ba.bitarray([1, 0, 0])]
#
#         assert node_rrl.last_cut_added_id == 2
#         assert node_rrl.last_cut_added_orientation is True
#         assert node_rrl.tangle.specification == {0: False, 1: False, 2: True}
#
#         assert node_rrl.tangle.cuts == [ba.bitarray([1, 0, 1]), ba.bitarray([0, 1, 1]), ba.bitarray([0, 0, 1])]
#         assert node_rrl.tangle.core == [ba.bitarray([0, 0, 1])]
#
#     def test_tree_points_case_permutation_three(self):
#         values = np.array([[0, 0, 1],
#                            [0, 1, 0],
#                            [1, 0, 0]]).astype(bool)
#         costs = np.array([1, 1, 1])
#         cuts = Cuts(values=values, costs=costs)
#
#         agreement = 1
#
#         tangle_tree = tangle_computation(cuts=cuts, agreement=agreement, verbose=0)
#         assert tangle_tree.__class__ == TangleTree
#
#         assert len(tangle_tree.maximals) == 3
#
#         node_lrr = tangle_tree.maximals[0]
#         node_rlr = tangle_tree.maximals[1]
#         node_rrl = tangle_tree.maximals[2]
#
#         assert node_lrr == tangle_tree.root.left_child.right_child.right_child
#         assert node_rlr == tangle_tree.root.right_child.left_child.right_child
#         assert node_rrl == tangle_tree.root.right_child.right_child.left_child
#
#         assert node_lrr.last_cut_added_id == 2
#         assert node_lrr.last_cut_added_orientation is False
#         assert node_lrr.tangle.specification == {0: True, 1: False, 2: False}
#
#         assert node_lrr.tangle.cuts == [ba.bitarray([0, 0, 1]), ba.bitarray([1, 0, 1]), ba.bitarray([0, 1, 1])]
#         assert node_lrr.tangle.core == [ba.bitarray([0, 0, 1])]
#
#         assert node_rlr.last_cut_added_id == 2
#         assert node_rlr.last_cut_added_orientation is False
#         assert node_rlr.tangle.specification == {0: False, 1: True, 2: False}
#
#         assert node_rlr.tangle.cuts == [ba.bitarray([1, 1, 0]), ba.bitarray([0, 1, 0]), ba.bitarray([0, 1, 1])]
#         assert node_rlr.tangle.core == [ba.bitarray([0, 1, 0])]
#
#         assert node_rrl.last_cut_added_id == 2
#         assert node_rrl.last_cut_added_orientation is True
#         assert node_rrl.tangle.specification == {0: False, 1: False, 2: True}
#
#         assert node_rrl.tangle.cuts == [ba.bitarray([1, 1, 0]), ba.bitarray([1, 0, 1]), ba.bitarray([1, 0, 0])]
#         assert node_rrl.tangle.core == [ba.bitarray([1, 0, 0])]

class Test_tangle_tree_node:

    def test_init(self):

        root = TangleNode(None, None, None, None, None, None, 0, None, Tangle())

        assert root.is_leaf() is True


class Test_tangle_tree:

    EXAMPLE1 = {'cuts': Cuts(values=np.array([[1, 1, 1, 1, 0, 0, 0, 0],
                                              [1, 1, 1, 0, 0, 0, 0, 0],
                                              [1, 1, 0, 0, 1, 1, 0, 0],
                                              [1, 0, 1, 0, 1, 0, 1, 0]], dtype=bool),
                             costs=[0.1, 0.5, 1, 1.5]),
                'agreement': 2}

    EXAMPLE2 = {'cuts': Cuts(values=np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                              [1, 1, 1, 0, 1, 1, 1, 1, 0, 0],
                                              [1, 1, 1, 1, 1, 1, 0, 0, 1, 0]], dtype=bool),
                             costs=[0.1, 0.5, 1]),
                'agreement': 2}

    def test_adding(self):

        tree = TangleTree(self.EXAMPLE1['agreement'])

        assert tree.root.is_leaf() is True

        tree.add_cut(self.EXAMPLE1['cuts'].values[0], 0)

        assert tree.root.is_leaf() is False
        assert len(tree.active) == 2

        tree.add_cut(self.EXAMPLE1['cuts'].values[1], 1)

        assert len(tree.active) == 2

        tree.add_cut(self.EXAMPLE1['cuts'].values[2], 2)

        assert len(tree.active) == 3

        tree.add_cut(self.EXAMPLE1['cuts'].values[3], 3)

        assert len(tree.maximals) == 3
        assert len(tree.active) == 0

    def test_contraction1(self):

        tree = TangleTree(self.EXAMPLE1['agreement'])

        core_algorithm(tree, self.EXAMPLE1['cuts'].values, np.arange(4))

        assert len(tree.maximals) == 3

        contracted_tree1 = ContractedTangleTree(tree, 0)

        contracted_tree2 = ContractedTangleTree(tree, 1)

        assert len(contracted_tree1.maximals) == len(tree.maximals)

        assert contracted_tree2.root.left_child.is_leaf() is True
        assert contracted_tree2.root.right_child.is_leaf() is True

    def test_contraction2(self):

        tree = TangleTree(self.EXAMPLE2['agreement'])

        core_algorithm(tree, self.EXAMPLE2['cuts'].values, np.arange(3))

        assert len(tree.maximals) + len(tree.active) == 4

        contracted_tree1 = ContractedTangleTree(tree, 0)

        contracted_tree2 = ContractedTangleTree(tree, 1)

        assert len(contracted_tree1.maximals) == len(tree.maximals) + len(tree.active)

        assert contracted_tree2.root.is_leaf() is True


