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
        if node.valid:
            if node.left is None:
                node.left = TangleTreeNode(node, c, False)
                self.check_validity(node.left)
            else:
                self.add_cut(node.left, c)

            if node.right is None:
                node.right = TangleTreeNode(node, c, True)
                self.check_validity(node.right)
            else:
                self.add_cut(node.right, c)
        return

    # checks for a single node if it is consistent (cuts of each triplet contains more points than agreement)
    def check_validity(self, node):
        nb_core_cuts = len(node.core_cuts)
        node.valid = True
        for i in range(nb_core_cuts):
            if sum(node.core_cuts[i]) < self.agreement:
                node.valid = False
                return
            for j in range(i+1, nb_core_cuts):
                if sum([a and b
                        for a, b in zip(node.core_cuts[i],
                                        node.core_cuts[j])]) < self.agreement:
                    node.valid = False
                    return
                for k in range(j+1, nb_core_cuts):
                    if sum([a and b and c
                            for a, b, c in zip(node.core_cuts[i],
                                             node.core_cuts[j],
                                             node.core_cuts[k])]) < self.agreement:
                        node.valid = False
                        return

        return

    def print(self, condensed=False):
        if condensed:
            self.condensed_tree.traverse_print(self.condensed_tree.root)
        else:
            self.traverse_print(self.root)

    # traverse the tree in depth first search and print out coordinates and the cut
    def traverse_print(self, node):
        if node and node.valid:
            print(node.coordinate)
            self.traverse_print(node.right)
            self.traverse_print(node.left)
        return

    # calculates and sets the attributes if neseccary
    def check_for_splitting_tangles(self):
        if self.nb_splitting_tangles is None:
            number, coordinates = self.get_splitting_tangles(self.root)
            self.nb_splitting_tangles = number
            self.splitting_tangles = coordinates

    # calculates all splitting tangles
    def get_splitting_tangles(self, node):
        if node is not None and node.valid:
            if node.left and node.left.valid and node.right and node.right.valid or node.left is None and node.right is None:
                self.condensed_tree.add(node)
            right = self.get_splitting_tangles(node.right)
            left = self.get_splitting_tangles(node.left)
            if node.left and node.left.valid and node.right and node.right.valid:
                node.splitting_or_leaf = True
                return 1 + left[0] + right[0], [node.coordinate] + left[1] + right[1]
            if node.left is None and node.right is None:
                node.splitting_or_leaf = True

            else:
                return left[0] + right[0], left[1] + right[1]
        return 0, []

    def get_leaves(self, node):
        if node.valid and not node.left and not node.right:
            return [node.coordinate]
        elif node.valid:
            return self.get_leaves(node.left) + self.get_leaves(node.right)

        return []

    def update_p(self, node=-1, right=None):
        if node:
            if node == -1:
                node = self.condensed_tree.root
            self.calculate_p(node, right)
            self.update_p(node.left, False)
            self.update_p(node.right, True)


    def calculate_p(self, node, right):
        if node.right and node.left:
            node.p = np.zeros(len(self.cuts[0]))

            coord_left = node.left.coordinate
            coord_right = node.right.coordinate

            print("left", coord_left)
            print("right", coord_right)

            if len(coord_left) >= len(coord_right):
                idx = np.arange(len(coord_left))
                idx = idx[len(node.coordinate):len(coord_right)]

                coord_left = coord_left[len(node.coordinate):len(coord_right)]
                coord_right = coord_right[len(node.coordinate):]
            else:
                idx = np.arange(len(coord_right))
                idx = idx[len(node.coordinate):len(coord_left)]

                coord_left = coord_left[len(node.coordinate):]
                coord_right = coord_right[len(node.coordinate):len(coord_left)]

            bool = np.not_equal(coord_left, coord_right)

            for b, i in zip(bool, idx):
                if b:
                    node.p += self.cuts[i]
                else:
                    node.p += np.flip(self.cuts[i])

            node.p = node.p / sum(bool)

            if node.parent:
                if right:
                    node.p = np.multiply(node.parent.p, node.p)
                else:
                    node.p = np.multiply((1 - node.parent.p), node.p)
        else:
            if right:
                node.p = node.parent.p
            else:
                if node.parent.parent:
                    node.p = node.parent.parent.p - node.parent.p
                else:
                    node.p = 1 - node.parent.p

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
            self.valid = True
            self.coordinate = []
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

    def add(self, treenode):
        #print("want to add the node")
        #print(treenode.coordinate)
        if self.root:
            self.add_node(self.root, treenode)
        else:
            #print("adding the node \n")
            self.root = CondensedTangleTreeNode(None, treenode)

    def add_node(self, node, treenode):
        #print("parent: ", node.coordinate)
        coord_node = node.coordinate
        coord_treenode = treenode.coordinate

        if coord_treenode[:len(coord_node)] != coord_node:
            #print("you went to far")
            return

        if node.left is not None:
            #print("going left \n")
            self.add_node(node.left, treenode)
        if node.right is not None:
            #print("going right \n")
            self.add_node(node.right, treenode)

        if not node.left or not node.right:
            #print("checking the sides")
            if coord_treenode[len(coord_node)]:
                #print("adding the node right \n")
                node.right = CondensedTangleTreeNode(node, treenode)
            else:
                #print("adding the node left \n")
                node.left = CondensedTangleTreeNode(node, treenode)

    # traverse the tree in depth first search and print out coordinates and the cut
    def traverse_print(self, node):
        if node:
            print(node.coordinate, node.p)
            self.traverse_print(node.right)
            self.traverse_print(node.left)


class CondensedTangleTreeNode:

    def __init__(self, parent, treenode):
        self.treenode = treenode

        self.parent = parent
        self.left = None
        self.right = None

        self.coordinate = treenode.coordinate

        self.p = None




# checks if the oriented cut a is a subset of the oriented cut b
def subset(a, b):
    return sum([x and y for x, y in zip(a, b)]) == sum(b)


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
