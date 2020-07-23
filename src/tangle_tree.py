from copy import deepcopy
import numpy as np
import bitarray as ba
#from bitarray.util import subset


def one(_):
    return 1

class TangleTreeModel:
    def __init__(self, agreement, cuts, costs=None, weight_fun=None, print_cuts=False):
        self.agreement = agreement
        self.cuts = cuts

        self.nb_points = len(self.cuts[0])
        self.nb_cuts = len(self.cuts)

        self.costs = (costs - min(costs))/max(costs) if any(costs) else np.ones(len(cuts))
        self.weight_function = weight_fun if weight_fun else one

        self.tree = TangleTree()
        self.condensed_tree = CondensedTangleTree()

        self.tangles = []
        self.maximals = []

        self.build(print_cuts)

    def build(self, print_cuts):
        print(" --- Building the tree of cuts \n")
        self.build_tree(print_cuts)

        print(" --- Building the condensed tree \n")
        self.build_condensed_tree()

        print(" --- Found ", str(len(self.maximals)), " interesting tangles.")

    '''
    function to build the Tangle Tree 
    '''
    def build_tree(self, print_cuts):
        for i, c in enumerate(self.cuts):
            if print_cuts:
                print("adding the cut ", str(i))
            self.add_cut(c)

    # adds a node to the tree
    def add_cut(self, c):
        new_leaves = []
        for parent in self.tree.current_leaves:
            parent.left = TangleTreeNode(parent, c, False)
            self.set_valid(parent.left)
            if parent.left.valid:
                new_leaves += [parent.left]

            parent.right = TangleTreeNode(parent, c, True)
            self.set_valid(parent.right)
            if parent.right.valid:
                new_leaves += [parent.right]
        self.tree.current_leaves = new_leaves

    # checks for a single node if it is consistent (cuts of each triplet contains more points than agreement)
    def set_valid(self, node):
        nb_core_cuts = len(node.core_cuts)
        node.valid = True
        for i in range(nb_core_cuts):
            if node.core_cuts[i].count() < self.agreement:
                node.valid = False
            for j in range(i + 1, nb_core_cuts):
                if (node.core_cuts[i] & node.core_cuts[j]).count() < self.agreement:
                    node.valid = False
                for k in range(j + 1, nb_core_cuts):
                    if (node.core_cuts[i] & node.core_cuts[j] & node.core_cuts[k]).count() < self.agreement:
                        node.valid = False
        return

    '''
    function to build the Condensed Tangle Tree 
    '''
    def build_condensed_tree(self):
        print("\t --- Calculating all the splitting tangles")
        self.get_splitting_tangles(self.tree.root)
        print("\t --- Calculating the set of usefull cuts for every splitting tangle")
        self.define_P_bottom_up()
        print("\t --- Calculating the probabilities for every splitting tangle and every leaf \n")
        self.calculate_p_top_down()

    # calculates all splitting tangles
    def get_splitting_tangles(self, node):
        if is_valid(node):
            if is_splitting(node) or is_leaf(node):
                node.splitting_or_leaf = True
                self.condensed_tree.add(node)
            self.get_splitting_tangles(node.right)
            self.get_splitting_tangles(node.left)

    def calculate_p_top_down(self, node=-1, side=None):
        if node:
            if node == -1:
                self.calculate_p_top_down(self.condensed_tree.root, None)
                return
            self.set_p(node, side)
            self.calculate_p_top_down(node.right, True)
            self.calculate_p_top_down(node.left, False)

    def set_p(self, node, side=None):
        if node.left and node.right:
            node.p_right = np.zeros(self.nb_points)
            node.p_left = np.zeros(self.nb_points)

            relevant = (node.right.P + node.left.P == 1)

            idx = np.arange(len(self.cuts))[len(node.coordinate):][relevant[len(node.coordinate):]]

            sides = node.right.P[idx]

            normalize = 0

            for i, s in zip(idx, sides):
                cost = self.weight_function(self.costs[i])

                if s:
                    node.p_right += self.cuts[i] * cost
                else:
                    node.p_right += np.flip(self.cuts[i]) * cost

                normalize += cost

            if normalize > 0:
                node.p_right = node.p_right / normalize

            if node.parent:
                if side:
                    node.p = node.parent.p_right
                else:
                    node.p = node.parent.p_left
            else:
                node.p = np.ones(len(self.cuts[0]))

            node.p_right = np.multiply(node.p, node.p_right)
            node.p_left = node.p - node.p_right

            self.tangles += [[node.p, node.coordinate, node.condensed_coordinate]]

        else:

            if node.condensed_coordinate:
                if side:
                    node.p = node.parent.p_right
                else:
                    node.p = node.parent.p_left
                self.tangles += [[node.p, node.coordinate, node.condensed_coordinate]]
                self.maximals += [[node.p, node.condensed_coordinate]]
            else:
                Warning("No tangles just one big cluster!")
                self.tangles += [[None, node.coordinate, node.condensed_coordinate]]

    def define_P_bottom_up(self):
        self.set_P(self.condensed_tree.root)

    def set_P(self, node):
        if node:
            if node.right and node.left:
                self.set_P(node.left)
                self.set_P(node.right)
                left_P = node.left.P
                right_P = node.right.P
                node.P = np.array([a if a - b == 0 else -1 for a, b in zip(left_P, right_P)], dtype=int)
            elif not node.left and not node.right:
                orientation = node.coordinate
                node.P = np.full(len(self.cuts), -1)
                node.P[:len(orientation)] = np.array(orientation, dtype=int)
            else:
                Warning("This should not happen!")

    '''
    additional functions
    '''
    def print(self, condensed=False):
        if condensed:
            self.condensed_tree.traverse_print(self.condensed_tree.root)
        else:
            self.tree.traverse_print(self.tree.root)


class TangleTree:
    def __init__(self):
        self.root = TangleTreeNode(None, None, None)
        self.current_leaves = [self.root]

    # traverse the tree in depth first search and print out coordinates and the cut
    def traverse_print(self, node):
        if is_valid(node):
            print(node.coordinate)
            self.traverse_print(node.right)
            self.traverse_print(node.left)


class TangleTreeNode:

    def __init__(self, parent, c, orientation):
        # cuts in the core of the tangle
        self.core = None
        self.core_cuts = []

        # connections to children and parent nodes
        self.parent = parent
        self.left = None
        self.right = None
        # indicates if orientation is consistent (valid) or not
        self.valid = False

        # if node is the root
        if parent is None:
            self.coordinate = []
            self.valid = True
            self.oriented_cut = None
        # else add the cut and update the core
        else:
            self.oriented_cut = ba.bitarray(list(c)) if orientation else ba.bitarray(list(~c))
            self.update_core()
            self.coordinate = parent.coordinate + [orientation]

    def update_core(self):
        core_cuts = deepcopy(self.parent.core_cuts)
        core = self.parent.core

        if core is None:
            self.core = deepcopy(self.oriented_cut)
            self.core_cuts += [deepcopy(self.oriented_cut)]
        else:

            if subset(self.oriented_cut, core):
                self.core = deepcopy(self.oriented_cut)
                self.core_cuts = [deepcopy(self.oriented_cut)]
            elif subset(core, self.oriented_cut):
                self.core = core
                self.core_cuts = core_cuts
            else:
                self.core = deepcopy(core & self.oriented_cut)
                delete = []
                for i, c in enumerate(core_cuts):
                    if subset(self.oriented_cut, c):
                        delete += [i]
                for i in np.flip(delete):
                    del core_cuts[i]

                core_cuts += [deepcopy(self.oriented_cut)]

                self.core_cuts = core_cuts



class CondensedTangleTree:

    # treenode refers to a node in the original tree while node is the node in the condensed tree

    def __init__(self):
        self.root = None

    def add(self, treenode):
        #print("want to add the node")
        #print(treenode.coordinate)
        if self.root:
            self.add_node(self.root, treenode)
        else:
            self.root = CondensedTangleTreeNode(None, treenode, None)

    def add_node(self, node, treenode):
        coord_node = node.coordinate
        coord_treenode = treenode.coordinate

        if coord_treenode[:len(coord_node)] != coord_node:
            return

        if coord_treenode[len(coord_node)]:
            # going to right side
            if node.right:
                if len(node.right.coordinate) < len(treenode.coordinate):
                    self.add_node(node.right, treenode)
                else:
                    Warning("something weird happened, think again! (right)")
            else:
                node.right = CondensedTangleTreeNode(node, treenode, True)
        else:
            # going to left side
            if node.left:
                if len(node.left.coordinate) < len(treenode.coordinate):
                    self.add_node(node.left, treenode)
                else:
                    Warning("something weird happened, think again! (left)")
            else:
                node.left = CondensedTangleTreeNode(node, treenode, False)

    # traverse the tree in depth first search and print out coordinates and the cut
    def traverse_print(self, node):
        if node:
            print(node.coordinate, node.p)
            self.traverse_print(node.right)
            self.traverse_print(node.left)


class CondensedTangleTreeNode:

    def __init__(self, parent, treenode, side):
        self.treenode = treenode

        # remember if node is left of right child of parent node
        self.side = side

        # if node is not the root, add the orientation to the orientation of the parent node
        if parent:
            self.condensed_coordinate = parent.condensed_coordinate + [side]
        else:
            self.condensed_coordinate = []

        self.depth = len(self.condensed_coordinate)

        # link to parent and children
        self.parent = parent
        self.left = None
        self.right = None

        # orientation of the cuts in the original tree
        self.coordinate = treenode.coordinate

        # to calculate the probabilities

        # list of cuts that are relevant for calculation
        self.P = None

        # probability of ending up in this tangle
        self.p = None
        # probability for being put to the left/right branch
        self.p_right = None
        self.p_left = None


# checks if the oriented cut a is a subset of the oriented cut b
def subset(a, b):
    return (a & b).count() == a.count()

# just try to reduce the typing and make it more readable
def is_splitting(node):
    return (is_valid(node.left) and is_valid(node.right))

def is_leaf(node):
    return (not node.left or not node.left.valid) and (not node.right or not node.right.valid)

def is_valid(node):
    return node is not None and node.valid
