import numpy as np


class Node:
    """
    Class to represent the nested components of a set of cuts.
    For every node we store the largest subset we found in self.subsets and then we store the
    first incompatible in self.components.

    Note that the structure of the tree is dependent on the order in which we add the cuts
    but it does not matter for the correctness of the computation.

    TODO: Check if there is a way to keep it balanced.
    """
    subsets = None
    components = None
    leaf = True

    def __init__(self, partition=None, complement_partition=None):

        self.partition = partition
        self.complement_partition = complement_partition

    def to_numpy(self, n_cuts):
        cut = np.zeros(n_cuts, dtype=bool)
        cut[list(self.partition)] = True

        return cut

    def insert(self, new_partition, new_complement_partition):
        """
        Add a new cut to a node

        Parameters
        ----------
        new_cut: array of shape [n_points]
            the new cut to add
        i_new_cut: int
            the index of the new cut in the set of all cuts

        Returns
        -------
            node: Node
                the new root of the tree after we added the node
            flip_cut: bool
                True if we had to flip the cut in order to insert it in the tree,
                False otherwise
        """

        # Root case
        if self.partition is None:
            if self.subsets is None:
                new_node = Node(new_partition, new_complement_partition)
                self.subsets = new_node
                return self
            else:
                self.subsets.insert(new_partition, new_complement_partition)
                return self

        is_new_subset = False
        is_new_superset = False

        if new_partition.issubset(self.partition):
            is_new_subset = True
            complement = False
        elif new_complement_partition.issubset(self.partition):
            is_new_subset = True
            complement = True
        elif self.partition.issubset(new_partition):
            is_new_superset = True
            complement = False
        elif self.partition.issubset(new_complement_partition):
            is_new_superset = True
            complement = True

        if is_new_subset:
            if self.subsets is None:
                if not complement:
                    new_node = Node(new_partition, new_complement_partition)
                else:
                    new_node = Node(new_complement_partition, new_partition)
                self.subsets = new_node
                self.leaf = False
            else:
                if not complement:
                    self.subsets = self.subsets.insert(new_partition, new_complement_partition)
                else:
                    self.subsets = self.subsets.insert(new_complement_partition, new_partition)
            return self

        elif is_new_superset:

            if not complement:
                new_node = Node(new_partition, new_complement_partition)
            else:
                new_node = Node(new_complement_partition, new_partition)
            new_node.subsets = self
            new_node.leaf = False

            # Check if something in the incompatible must be moved to parent
            node = self.components
            previous_child = self
            last_parent = new_node

            while node is not None:
                if not new_node.partition.issuperset(node.partition):

                    # Move the node
                    last_parent.components = node

                    # Update pointers
                    last_parent = node
                    node = node.components

                    # Cut links
                    previous_child.components = node
                    last_parent.components = None

                else:
                    previous_child = previous_child.components
                    node = node.components

            return new_node
        else:
            self.leaf = False
            if self.components is None:
                self.components = Node(new_partition, new_complement_partition)
            else:
                self.components = self.components.insert(new_partition, new_complement_partition)
            return self


