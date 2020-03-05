import numpy as np


class Node:
    """
    Class to represent the nested components of a set of cuts.
    For every node we store the largest subset we found in self.subsets and then we store the
    first incompatible in self.incomps.

    Note that the structure of the tree is dependent on the order in which we add the cuts
    but it does not matter for the correctness of the computation.

    TODO: Check if there is a way to keep it balanced.
    """
    subsets = None
    incomps = None
    leaf = True

    def __init__(self, cut):

        self.cut = cut
        n = len(cut)
        self.idx = set(np.arange(n)[cut])

    def insert(self, new_cut):
        """
        Add a new cut to a node

        Parameters
        ----------
        new_cut: array of shape [n_points]
            the new cut to add

        Returns
        -------
            node: Node
                the new root of the tree after we added the node
            flip_cut: bool
                True if we had to flip the cut in order to insert it in the tree,
                False otherwise
        """
        n = len(new_cut)
        idx = np.arange(n)
        new_idx, new_comp_idx = set(idx[new_cut]), set(idx[~new_cut])

        is_new_subset = False
        is_new_superset = False

        flip_cut = False
        if new_idx.issubset(self.idx):
            is_new_subset = True
        elif new_comp_idx.issubset(self.idx):
            new_cut = ~new_cut
            flip_cut = True
            is_new_subset = True
        elif new_idx.issuperset(self.idx):
            is_new_superset = True
        elif new_comp_idx.issuperset(self.idx):
            new_cut = ~new_cut
            flip_cut = True
            is_new_superset = True

        if is_new_subset:
            if self.subsets is None:
                self.subsets = Node(new_cut)
            else:
                self.subsets, _ = self.subsets.insert(new_cut)
            self.leaf = False
            return self, flip_cut

        elif is_new_superset:

            new_node = Node(new_cut)
            new_node.subsets = self
            new_node.leaf = False

            # Check if something in the incompatible must be moved to parent
            node = self.incomps
            previous_child = self
            last_parent = new_node

            while node is not None:
                if not new_node.idx.issuperset(node.idx):

                    # Move the node
                    last_parent.incomps = node

                    # Update pointers
                    last_parent = node
                    node = node.incomps

                    # Cut links
                    previous_child.incomps = node
                    last_parent.incomps = None

                else:
                    previous_child = previous_child.incomps

            return new_node, flip_cut
        else:
            self.leaf = False
            if self.incomps is None:
                self.incomps = Node(new_cut)
            else:
                self.incomps, _ = self.incomps.insert(new_cut)
            return self, flip_cut
