from copy import deepcopy
import numpy as np
from pythonlangutil.overload import Overload, signature


class TangleTree:
    def __init__(self, agreement, cuts):
        self.cuts = cuts
        self.agreement = agreement
        self.root = TangleTreeNode(None, None, None)
        self.build()
        self.nb_splitting_tangles = None
        self.splitting_tangles = None
        self.copy = False
        self.condensed_tree = CondensedTangleTree()

    # builds the tree by going trough all the cuts and adds them in the right order (given by order function)
    def build(self):
        for c in self.cuts:
            self.add_cut(self.root, c)

    # adds a node to the tree
    def add_cut(self, node, c):
        print("adding cut: ", c)
        if node.left is None:
            node.left = TangleTreeNode(node, c, False)
            self.is_valid(node.left)
        elif node.left.valid:
            self.add_cut(node.left, c)

        if node.right is None:
            node.right = TangleTreeNode(node, c, True)
            self.is_valid(node.right)
        elif node.right.valid:
            self.add_cut(node.right, c)

    # checks for a single node if it is consistent (cuts of each triplet contains more points than agreement)
    def is_valid(self, node):
        nb_core_cuts = len(node.core_cuts)
        node.valid = True
        for i in range(nb_core_cuts):
            if sum(node.core_cuts[i]) < self.agreement:
                node.valid = False
            for j in range(i+1, nb_core_cuts):
                if sum([a and b
                        for a, b in zip(node.core_cuts[i],
                                        node.core_cuts[j])]) < self.agreement:
                    node.valid = False
                for k in range(j+1, nb_core_cuts):
                    if sum([a and b and c
                            for a, b, c in zip(node.core_cuts[i],
                                             node.core_cuts[j],
                                             node.core_cuts[k])]) < self.agreement:
                        node.valid = False

        return

    def print(self, condensed=False):
        if condensed:
            self.condensed_tree.traverse_print(self.condensed_tree.root)
        else:
            self.traverse_print(self.root)

    # traverse the tree in depth first search and print out coordinates and the cut
    def traverse_print(self, node):
        if check(node):
            print(node.coordinate)
            self.traverse_print(node.right)
            self.traverse_print(node.left)

    # calculates and sets the attributes if neseccary
    def check_for_splitting_tangles(self):
        if self.nb_splitting_tangles is None:
            number, coordinates = self.get_splitting_tangles(self.root)
            self.nb_splitting_tangles = number
            self.splitting_tangles = coordinates

    # calculates all splitting tangles
    def get_splitting_tangles(self, node):
        if check(node):
            if (check(node.left) and check(node.right)) or ((not node.left or not node.left.valid) and (not node.right or not node.right.valid)):
                if not node.left and not node.right:
                    print("'I'm a leaf:")
                    print(node.left)
                    print(node.right)
                elif node.left.valid and node.right.valid:
                    print("I'm a splitting tangle")
                    print(node.coordinate)
                    print(node.left.valid)
                    print(node.right.valid)
                else:
                    print("i'm something weird")
                node.splitting_or_leaf = True
                self.condensed_tree.add(node)
            right = self.get_splitting_tangles(node.right)
            left = self.get_splitting_tangles(node.left)

            if check(node.left) and check(node.right):
                return 1 + left[0] + right[0], [node.coordinate] + left[1] + right[1]
            else:
                return left[0] + right[0], left[1] + right[1]
        return 0, []

    def get_leaves(self, node):
        if check(node):
            if (not node.left or not node.left.valid) and (not node.right or not node.right.valid):
                return [node.coordinate]
            else:
                return self.get_leaves(node.right) + self.get_leaves(node.left)
        else:
            return []

    def update_p(self, node=-1, side=None):
        if node:
            if node == -1:
                node = self.condensed_tree.root
                node.p = np.ones(len(self.cuts[0]))

            self.calculate_p(node, side)
            self.update_p(node.right, True)
            self.update_p(node.left, False)

    def calculate_p(self, node, side):
        if node.right and node.left:
            node.p_right = np.zeros(len(self.cuts[0]))
            node.p_left = np.zeros(len(self.cuts[0]))

            coord_left = np.array(node.left.coordinate)
            coord_right = np.array(node.right.coordinate)

            if len(coord_left) >= len(coord_right):
                idx = np.arange(len(coord_left))
                idx = idx[len(node.coordinate):len(coord_right)]
            else:
                idx = np.arange(len(coord_right))
                idx = idx[len(node.coordinate):len(coord_left)]

            idx = idx[np.not_equal(coord_left[idx], coord_right[idx])]

            sides = coord_right[idx]
            print(idx)

            for i, s in zip(idx, sides):
                if s:
                    node.p_right += self.cuts[i]
                else:
                    node.p_right += np.flip(self.cuts[i])

            if len(idx) > 1:
                node.p_right = node.p_right / len(idx)

            if node.parent:
                if side:
                    node.p = node.parent.p_right
                else:
                    node.p = node.parent.p_left
            else:
                node.p = np.ones(len(self.cuts[0]))

            node.p_right = np.multiply(node.p, node.p_right)
            node.p_left = node.p - node.p_right

            self.condensed_tree.tangles += [[list(node.p), node.coordinate, node.condensed_coordinate]]
        else:
            print("I'm a leaf")
            if node.condensed_coordinate:
                print("Coordinate: ", node.condensed_coordinate)
                if side:
                    node.p = node.parent.p_right
                else:
                    node.p = node.parent.p_left

            else:
                self.condensed_tree.tangles += [[None, node.coordinate, node.condensed_coordinate]]


class TangleTreeNode:

    def __init__(self, parent, c, orientation):
        self.core = None
        self.core_cuts = []

        self.parent = parent
        self.left = None
        self.right = None
        self.valid = False

        self.splitting_or_leaf = False

        if parent is None:
            self.coordinate = []
            self.valid = True
            self.oriented_cut = None
        else:
            self.oriented_cut = c if orientation else [not x for x in c]
            self.update_core()
            self.coordinate = parent.coordinate + [orientation]

    def update_core(self):
        core = deepcopy(self.parent.core)
        core_cuts = deepcopy(self.parent.core_cuts)

        if core is None:
            self.core = deepcopy(self.oriented_cut)
            self.core_cuts += [deepcopy(self.oriented_cut)]
        else:

            if subset(self.oriented_cut, core):
                self.core = core
                self.core_cuts = core_cuts
            elif subset(core, self.oriented_cut):
                self.core = deepcopy(self.oriented_cut)
                self.core_cuts = [deepcopy(self.oriented_cut)]
            else:
                delete = []
                for i, c in enumerate(core_cuts):
                    if subset(c, self.oriented_cut):
                        delete += [i]
                for i in np.flip(delete):
                    del core_cuts[i]

                core_cuts += [self.oriented_cut]

                self.core = [x and y for x, y in zip(core, self.oriented_cut)]
                self.core_cuts = core_cuts


class CondensedTangleTree:

    # treenode refers to a node in the original tree while node is the node in the condensed tree

    def __init__(self):
        self.root = None
        self.tangles = []

    def add(self, treenode):
        #print("want to add the node")
        #print(treenode.coordinate)
        if self.root:
            self.add_node(self.root, treenode)
        else:
            #print("adding the node \n")
            self.root = CondensedTangleTreeNode(None, treenode, None)

    def add_node(self, node, treenode):
        #print("parent: ", node.coordinate)
        #print(treenode.coordinate)
        coord_node = node.coordinate
        coord_treenode = treenode.coordinate

        if coord_treenode[:len(coord_node)] != coord_node:
            #print("you went to far")
            return

        if coord_treenode[len(coord_node)]:
            # going to right side
            if node.right:
                if len(node.right.coordinate) < len(treenode.coordinate):
                    #print("going right \n")
                    self.add_node(node.right, treenode)
                else:
                    print("something weird happened, think again! (right)")
            else:
                node.right = CondensedTangleTreeNode(node, treenode, True)
                #print("added the node right \n")
        else:
            # going to left side
            if node.left:
                if len(node.left.coordinate) < len(treenode.coordinate):
                    #print("going left \n")
                    self.add_node(node.left, treenode)
                else:
                    print("something weird happened, think again! (left)")
            else:
                node.left = CondensedTangleTreeNode(node, treenode, False)
                #print("added the node left \n")

    # traverse the tree in depth first search and print out coordinates and the cut
    def traverse_print(self, node):
        if node:
            print(node.coordinate, node.p)
            self.traverse_print(node.right)
            self.traverse_print(node.left)


class CondensedTangleTreeNode:

    def __init__(self, parent, treenode, side):
        self.treenode = treenode

        self.side = side

        if parent:
            self.condensed_coordinate = parent.condensed_coordinate + [side]
        else:
            self.condensed_coordinate = []

        self.parent = parent
        self.left = None
        self.right = None

        self.coordinate = treenode.coordinate

        self.p = None
        self.p_right = None
        self.p_left = None


# checks if the oriented cut a is a subset of the oriented cut b
def subset(a, b):
    return sum([x and y for x, y in zip(a, b)]) == sum(b)


def check(node):
    return node is not None and node.valid

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
