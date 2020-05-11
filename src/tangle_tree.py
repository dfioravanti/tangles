from copy import deepcopy
import numpy as np


class TangleTreeNode:
    def __init__(self, parent, c, orientation):
        self.core = None
        self.core_cuts = []

        self.parent = parent
        self.left = None
        self.right = None

        self.valid = False
        self.is_splitting_tangle = False

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


class TangleTree:
    def __init__(self, agreement, cuts):
        self.cuts = cuts
        self.agreement = agreement
        self.root = TangleTreeNode(None, None, None)
        self.build()
        self.nb_splitting_tangles = None
        self.splitting_tangles = None
        self.copy = False

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
            right = self.get_splitting_tangles(node.right)
            left = self.get_splitting_tangles(node.left)
            if node.left is not None and node.left.valid and node.right is not None and node.right.valid:
                node.is_splitting_tangle = True
                return 1 + left[0] + right[0], [node.coordinate] + left[1] + right[1]
            else:
                return left[0] + right[0], left[1] + right[1]
        return 0, []

    def hard_clustering(self):
        nb_clusters = self.nb_splitting_tangles + 1
        nb_cuts = len(self.cuts)
        splitting = self.splitting_tangles
        leaves = self.get_leaves(self.root)

    def get_leaves(self, node):
        if node.valid and not node.left and not node.right:
            return [node.coordinate]
        elif node.valid:
            return self.get_leaves(node.left) + self.get_leaves(node.right)

        return []

    def first_splitting_tangle(self):
        node = self.root
        first_tangle = self.splitting_tangles[0]
        for bool in first_tangle:
            if bool:
                node = node.right
            else:
                node = node.left
        return node

#checks if the oriented cut a is a subset of the oriented cut b
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
