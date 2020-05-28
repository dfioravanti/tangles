import sys
sys.path.append("../src")

import bitarray as ba
import numpy as np

from src.execution import tangle_computation
from src.tree_tangles import TangleTree

class Test_tangle_computation(object):

    def test_one_point_case(self):

        cuts_values = np.zeros((1, 1), dtype=bool)
        orders = np.zeros(1, dtype=float)

        cuts_values[0, 0] = True
        orders[0] = 1
        agreement = 1

        cuts = {}
        cuts['values'] = cuts_values

        tangle_tree = tangle_computation(cuts=cuts, orders=orders, agreement=1, verbose=0)
        assert tangle_tree.__class__ == TangleTree

        assert tangle_tree.root.left_child is not None
        assert tangle_tree.root.right_child is None

        node = tangle_tree.root.left_child 
        assert node.last_cut_added_id == 0
        assert node.last_cut_added_orientation == cuts_values[0]
        assert node.tangle.specification == {0: True}
        assert node.tangle.cuts == [ba.bitarray(cuts_values.tolist())]
        assert node.tangle.core == [ba.bitarray(cuts_values.tolist())]


        cuts_values[0, 0] = False
        orders[0] = 1
        agreement = 1

        cuts = {}
        cuts['values'] = cuts_values

        tangle_tree = tangle_computation(cuts=cuts, orders=orders, agreement=1, verbose=0)
        assert tangle_tree.__class__ == TangleTree

        assert tangle_tree.root.left_child is None
        assert tangle_tree.root.right_child is not None

        node = tangle_tree.root.right_child 
        assert node.last_cut_added_id == 0
        assert node.last_cut_added_orientation == cuts_values[0]
        assert node.tangle.specification == {0: False}
        assert node.tangle.cuts == [ba.bitarray(cuts_values.tolist())]
        assert node.tangle.core == [ba.bitarray(cuts_values.tolist())]

        cuts_values = np.zeros((2, 1), dtype=bool)
        orders = np.zeros(2, dtype=float)

        cuts_values[:, 0] = [True, False]
        orders[:] = [1, 1]
        agreement = 1

        cuts = {}
        cuts['values'] = cuts_values

        tangle_tree = tangle_computation(cuts=cuts, orders=orders, agreement=1, verbose=0)
        assert tangle_tree.__class__ == TangleTree

        assert tangle_tree.root.left_child is not None
        assert tangle_tree.root.right_child is None

        l_node = tangle_tree.root.left_child 
        assert l_node.last_cut_added_id == 0
        assert l_node.last_cut_added_orientation == cuts_values[0, :]
        assert l_node.tangle.specification == {0: True}
        assert l_node.tangle.cuts == [ba.bitarray(cuts_values[0].tolist())]
        assert l_node.tangle.core == [ba.bitarray(cuts_values[0].tolist())]

        assert l_node.left_child is None
        assert l_node.right_child is not None

        lr_node = l_node.right_child
        assert lr_node.last_cut_added_id == 1
        assert lr_node.last_cut_added_orientation == cuts_values[1, :]
        assert lr_node.tangle.specification == {0: True, 1: False}

        assert lr_node.tangle.cuts == [ba.bitarray([1]), ba.bitarray([1])]
        assert lr_node.tangle.core == [ba.bitarray([1])]

    def test_tree_points_case_permutation_one(self):

        cuts_values = np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0 ,1]]).astype(bool)
        orders = np.array([1, 1, 1])
        agreement = 1
        
        cuts = {}
        cuts['values'] = cuts_values

        tangle_tree = tangle_computation(cuts=cuts, orders=orders, agreement=1, verbose=0)
        assert tangle_tree.__class__ == TangleTree
 
        assert len(tangle_tree.maximals) == 3
        
        node_lrr = tangle_tree.maximals[0]
        node_rlr = tangle_tree.maximals[1]
        node_rrl = tangle_tree.maximals[2]

        assert node_lrr == tangle_tree.root.left_child.right_child.right_child
        assert node_rlr == tangle_tree.root.right_child.left_child.right_child
        assert node_rrl == tangle_tree.root.right_child.right_child.left_child
        
        assert node_lrr.last_cut_added_id == 2
        assert node_lrr.last_cut_added_orientation == False
        assert node_lrr.tangle.specification == {0: True, 1: False, 2: False}

        assert node_lrr.tangle.cuts == [ba.bitarray([1, 0, 0]), ba.bitarray([1, 0, 1]), ba.bitarray([1, 1, 0])]
        assert node_lrr.tangle.core == [ba.bitarray([1, 0, 0])]
        
        assert node_rlr.last_cut_added_id == 2
        assert node_rlr.last_cut_added_orientation == False
        assert node_rlr.tangle.specification == {0: False, 1: True, 2: False}

        assert node_rlr.tangle.cuts == [ba.bitarray([0, 1, 1]), ba.bitarray([0, 1, 0]), ba.bitarray([1, 1, 0])]
        assert node_rlr.tangle.core == [ba.bitarray([0, 1, 0])]
        
        assert node_rrl.last_cut_added_id == 2
        assert node_rrl.last_cut_added_orientation == True
        assert node_rrl.tangle.specification == {0: False, 1: False, 2: True}

        assert node_rrl.tangle.cuts == [ba.bitarray([0, 1, 1]), ba.bitarray([1, 0, 1]), ba.bitarray([0, 0, 1])]
        assert node_rrl.tangle.core == [ba.bitarray([0, 0, 1])]
        
    def test_tree_points_case_permutation_two(self):

        cuts_values = np.array([[0, 1, 0],
                         [1, 0, 0],
                         [0, 0 ,1]]).astype(bool)
        orders = np.array([1, 1, 1])
        agreement = 1
    
        cuts = {}
        cuts['values'] = cuts_values

        tangle_tree = tangle_computation(cuts=cuts, orders=orders, agreement=1, verbose=0)
        assert tangle_tree.__class__ == TangleTree
 
        assert len(tangle_tree.maximals) == 3
        
        node_lrr = tangle_tree.maximals[0]
        node_rlr = tangle_tree.maximals[1]
        node_rrl = tangle_tree.maximals[2]

        assert node_lrr == tangle_tree.root.left_child.right_child.right_child
        assert node_rlr == tangle_tree.root.right_child.left_child.right_child
        assert node_rrl == tangle_tree.root.right_child.right_child.left_child
        
        assert node_lrr.last_cut_added_id == 2
        assert node_lrr.last_cut_added_orientation == False
        assert node_lrr.tangle.specification == {0: True, 1: False, 2: False}

        assert node_lrr.tangle.cuts == [ba.bitarray([0, 1, 0]), ba.bitarray([0, 1, 1]), ba.bitarray([1, 1, 0])]
        assert node_lrr.tangle.core == [ba.bitarray([0, 1, 0])]
        
        assert node_rlr.last_cut_added_id == 2
        assert node_rlr.last_cut_added_orientation == False
        assert node_rlr.tangle.specification == {0: False, 1: True, 2: False}

        assert node_rlr.tangle.cuts == [ba.bitarray([1, 0, 1]), ba.bitarray([1, 0, 0]), ba.bitarray([1, 1, 0])]
        assert node_rlr.tangle.core == [ba.bitarray([1, 0, 0])]
        
        assert node_rrl.last_cut_added_id == 2
        assert node_rrl.last_cut_added_orientation == True
        assert node_rrl.tangle.specification == {0: False, 1: False, 2: True}

        assert node_rrl.tangle.cuts == [ba.bitarray([1, 0, 1]), ba.bitarray([0, 1, 1]), ba.bitarray([0, 0, 1])]
        assert node_rrl.tangle.core == [ba.bitarray([0, 0, 1])]
        
    def test_tree_points_case_permutation_three(self):

        cuts_values = np.array([[0, 0, 1],
                         [0, 1, 0],
                         [1, 0, 0]]).astype(bool)
        orders = np.array([1, 1, 1])
        agreement = 1
        
        cuts = {}
        cuts['values'] = cuts_values

        tangle_tree = tangle_computation(cuts=cuts, orders=orders, agreement=1, verbose=0)
        assert tangle_tree.__class__ == TangleTree
 
        assert len(tangle_tree.maximals) == 3
        
        node_lrr = tangle_tree.maximals[0]
        node_rlr = tangle_tree.maximals[1]
        node_rrl = tangle_tree.maximals[2]

        assert node_lrr == tangle_tree.root.left_child.right_child.right_child
        assert node_rlr == tangle_tree.root.right_child.left_child.right_child
        assert node_rrl == tangle_tree.root.right_child.right_child.left_child
        
        assert node_lrr.last_cut_added_id == 2
        assert node_lrr.last_cut_added_orientation == False
        assert node_lrr.tangle.specification == {0: True, 1: False, 2: False}

        assert node_lrr.tangle.cuts == [ba.bitarray([0, 0, 1]), ba.bitarray([1, 0, 1]), ba.bitarray([0, 1, 1])]
        assert node_lrr.tangle.core == [ba.bitarray([0, 0, 1])]
        
        assert node_rlr.last_cut_added_id == 2
        assert node_rlr.last_cut_added_orientation == False
        assert node_rlr.tangle.specification == {0: False, 1: True, 2: False}

        assert node_rlr.tangle.cuts == [ba.bitarray([1, 1, 0]), ba.bitarray([0, 1, 0]), ba.bitarray([0, 1, 1])]
        assert node_rlr.tangle.core == [ba.bitarray([0, 1, 0])]
        
        assert node_rrl.last_cut_added_id == 2
        assert node_rrl.last_cut_added_orientation == True
        assert node_rrl.tangle.specification == {0: False, 1: False, 2: True}

        assert node_rrl.tangle.cuts == [ba.bitarray([1, 1, 0]), ba.bitarray([1, 0, 1]), ba.bitarray([1, 0, 0])]
        assert node_rrl.tangle.core == [ba.bitarray([1, 0, 0])]