from copy import deepcopy

import matplotlib.pyplot as plt

import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout, write_dot

import bitarray as ba
import numpy as np

from src.tangles import Tangle
from src.utils import matching_items, Orientation


class TangleNode(object):

    def __init__(self, parent, right_child, left_child, is_left_child, splitting,
                 did_split, last_cut_added_id, last_cut_added_orientation, tangle):

        self.parent = parent
        self.right_child = right_child
        self.left_child = left_child
        self.is_left_child = is_left_child

        self.splitting = splitting

        self.did_split = did_split
        self.last_cut_added_id = last_cut_added_id
        self.last_cut_added_orientation = last_cut_added_orientation

        self.tangle = tangle

    def __str__(self, height=0):

        if self.parent is None:
            string = 'Root'
        else:
            padding = ' '
            string = f'{padding * height}{self.last_cut_added_id} -> {self.last_cut_added_orientation}'

        if self.left_child is not None:
            string += '\n'
            string += self.left_child.__str__(height=height + 1)
        if self.right_child is not None:
            string += '\n'
            string += self.right_child.__str__(height=height + 1)

        return string


class ContractedTangleNode(TangleNode):

    def __init__(self, parent, node):

        attributes = node.__dict__
        super().__init__(**attributes)

        self.parent = parent
        self.right_child = None
        self.left_child = None

        self.characterizing_cuts = dict()
        self.characterizing_cuts_left = dict()
        self.characterizing_cuts_right = dict()

        self.is_left_child_processed = False
        self.is_right_child_processed = False

        self.p = None

    def __str__(self, height=0):
        string = ""

        if self.parent is None:
            string += 'Root\n'

        padding = '  '
        string_cuts = [f'{k} -> {v}' for k, v in self.characterizing_cuts_left.items()]
        string += f'{padding * height} left: {string_cuts}\n'

        string_cuts = [f'{k} -> {v}' for k, v in self.characterizing_cuts_right.items()]
        string += f'{padding * height} right: {string_cuts}'

        if self.left_child.left_child is not None:
            string += '\n'
            string += self.left_child.__str__(height=height + 1)
        if self.right_child.left_child is not None:
            string += '\n'
            string += self.right_child.__str__(height=height + 1)

        return string


# created new TangleNode and adds it as child to current node
def _add_new_child(current_node, tangle, last_cut_added_id, last_cut_added_orientation, did_split):
    new_node = TangleNode(parent=current_node,
                          right_child=None,
                          left_child=None,
                          is_left_child=last_cut_added_orientation,
                          splitting=False,
                          did_split=did_split,
                          last_cut_added_id=last_cut_added_id,
                          last_cut_added_orientation=last_cut_added_orientation,
                          tangle=tangle)

    if new_node.is_left_child:
        current_node.left_child = new_node
    else:
        current_node.right_child = new_node

    return new_node


class TangleTree(object):

    def __init__(self, agreement):

        self.root = TangleNode(parent=None,
                               right_child=None,
                               left_child=None,
                               splitting=None,
                               is_left_child=None,
                               did_split=None,
                               last_cut_added_id=None,
                               last_cut_added_orientation=None,
                               tangle=Tangle())
        self.active = [self.root]
        self.maximals = []
        self.will_split = []
        self.is_empty = True
        self.agreement = agreement

    def __str__(self):
        return str(self.root)

    # function to add a single cut to the tree
    # function checks if tree is empty
    def add_cut(self, cut, cut_id):
        current_active = self.active
        self.active = []

        could_add_one = False
        for current_node in current_active:
            could_add_node, did_split, is_maximal = self._add_children_to_node(current_node, cut, cut_id)
            could_add_one = could_add_one or could_add_node

            if did_split:
                current_node.splitting = True
                self.will_split.append(current_node)
            elif is_maximal:
                self.maximals.append(current_node)

        if could_add_one:
            self.is_empty = False

        return could_add_one

    def _add_children_to_node(self, current_node, cut, cut_id):
        old_tangle = current_node.tangle
        new_tangle_true = old_tangle.add(new_cut=ba.bitarray(cut.tolist()),
                                         new_cut_specification={cut_id: True},
                                         min_size=self.agreement)
        new_tangle_false = old_tangle.add(new_cut=ba.bitarray((~cut).tolist()),
                                          new_cut_specification={cut_id: False},
                                          min_size=self.agreement)

        could_add_one = False

        if new_tangle_true is not None and new_tangle_false is not None:
            did_split = True
        else:
            did_split = False

        if new_tangle_true is None and new_tangle_false is None:
            is_maximal = True
        else:
            is_maximal = False

        if new_tangle_true is not None:
            could_add_one = True
            new_node = _add_new_child(current_node=current_node,
                                      tangle=new_tangle_true,
                                      last_cut_added_id=cut_id,
                                      last_cut_added_orientation=True,
                                      did_split=did_split)
            self.active.append(new_node)

        if new_tangle_false is not None:
            could_add_one = True
            new_node = _add_new_child(current_node=current_node,
                                      tangle=new_tangle_false,
                                      last_cut_added_id=cut_id,
                                      last_cut_added_orientation=False,
                                      did_split=did_split)
            self.active.append(new_node)

        return could_add_one, did_split, is_maximal

    def plot_tree(self, node=None, depth=0, position=0, set_header=True, path=None):

        tree = nx.Graph()
        labels = self._add_node_to_nx(tree, self.root)

        pos = graphviz_layout(tree, prog='dot')

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
        nx.draw_networkx(tree, pos=pos, ax=ax, labels=labels, node_size=1500)
        plt.savefig(path)

    def _add_node_to_nx(self, tree, node, parent_id=None, direction=None):

        if node.parent is None:
            my_id = 'root'
            my_label = 'Root'

            tree.add_node(my_id)
        else:
            my_id = parent_id + direction
            str_o = 'T' if node.last_cut_added_orientation else 'F'
            my_label = f'{node.last_cut_added_id} -> {str_o}'

            tree.add_node(my_id)
            tree.add_edge(my_id, parent_id)

        labels = {my_id: my_label}

        if node.left_child is not None:
            left_labels = self._add_node_to_nx(tree, node.left_child, parent_id=my_id, direction='left')
            labels = {**labels, **left_labels}
        if node.right_child is not None:
            right_labels = self._add_node_to_nx(tree, node.right_child, parent_id=my_id, direction='right')
            labels = {**labels, **right_labels}

        return labels


class ContractedTangleTree(TangleTree):

    def __init__(self, tree):

        self.is_empty = tree.is_empty
        self.processed_soft_prediction = False
        self.maximals = []
        self.splitting = []
        self.root = self._contract_subtree(parent=None, node=tree.root)

    def __str__(self):
        return str(self.root)

    def _contract_subtree(self, parent, node):
        if node.left_child is None and node.right_child is None:
            # is leaf so create new node
            contracted_node = ContractedTangleNode(parent=parent, node=node)
            contracted_node.characterizing_cuts[node.last_cut_added_id] = Orientation(node.last_cut_added_orientation)
            self.maximals.append(contracted_node)
            return contracted_node
        elif node.left_child is not None and node.right_child is not None:
            # is splitting so create new node
            contracted_node = ContractedTangleNode(parent=parent, node=node)

            contracted_left_child = self._contract_subtree(parent=contracted_node, node=node.left_child)
            contracted_node.left_child = contracted_left_child
            contracted_node.is_left_child_processed = True

            contracted_right_child = self._contract_subtree(parent=contracted_node, node=node.right_child)
            contracted_node.right_child = contracted_right_child
            contracted_node.is_right_child_processed = True

            process_split(contracted_node)

            self.splitting.append(contracted_node)

            return contracted_node
        else:
            if node.left_child is not None:
                return self._contract_subtree(parent=parent, node=node.left_child)
            if node.right_child is not None:
                return self._contract_subtree(parent=parent, node=node.right_child)


def process_split(node):
    node_id = node.last_cut_added_id if node.last_cut_added_id else 0
    next_node_id = min(node.left_child.last_cut_added_id, node.right_child.last_cut_added_id)

    characterizing_cuts_left = node.left_child.characterizing_cuts
    characterizing_cuts_right = node.right_child.characterizing_cuts

    orientation_left = node.left_child.tangle.specification
    orientation_right = node.right_child.tangle.specification

    # add new relevant cuts
    for id_cut in range(node_id + 1, next_node_id + 1):
        characterizing_cuts_left[id_cut] = Orientation(orientation_left[id_cut])
        characterizing_cuts_right[id_cut] = Orientation(orientation_right[id_cut])

    id_not_in_both = (characterizing_cuts_left.keys() | characterizing_cuts_right.keys()) \
        .difference(characterizing_cuts_left.keys() & characterizing_cuts_right.keys())

    # if cuts are not oriented in both subtrees delete
    for id_cut in id_not_in_both:
        characterizing_cuts_left.pop(id_cut, None)
        characterizing_cuts_right.pop(id_cut, None)

    # characterizing cuts of the current node
    characterizing_cuts = {**characterizing_cuts_left, **characterizing_cuts_right}

    id_cuts_oriented_same_way = matching_items(characterizing_cuts_left, characterizing_cuts_right)

    # if they are oriented in the same way they are not relevant for distungishing but might be for 'higher' nodes
    # delete in the left and right parts but keep in the characteristics of the current node
    for id_cut in id_cuts_oriented_same_way:
        characterizing_cuts[id_cut] = characterizing_cuts_left[id_cut]
        characterizing_cuts_left.pop(id_cut)
        characterizing_cuts_right.pop(id_cut)

    id_cuts_oriented_both_ways = characterizing_cuts_left.keys() & characterizing_cuts_right.keys()

    # remove the cuts that are oriented in both trees but in different directions from the current node since they do
    # not affect higher nodes anymore
    for id_cut in id_cuts_oriented_both_ways:
        characterizing_cuts.pop(id_cut)

    node.characterizing_cuts_left = characterizing_cuts_left
    node.characterizing_cuts_right = characterizing_cuts_right
    node.characterizing_cuts = characterizing_cuts


def compute_soft_predictions_node(characterizing_cuts, cuts, costs, verbose):
    sum_p = np.zeros(len(cuts.values[0]))

    for i, o in characterizing_cuts.items():
        if o.direction == 'left':
            sum_p += np.array(cuts.values[i]) * costs[i]
        elif o.direction == 'right':
            sum_p += np.array(~cuts.values[i]) * costs[i]

    return sum_p


def compute_soft_predictions_children(node, cuts, costs, verbose):
    _, nb_points = cuts.values.shape

    if node.parent is None:
        node.p = np.ones(nb_points)

    if node.left_child is not None and node.right_child is not None:

        unnormalized_p_left = compute_soft_predictions_node(characterizing_cuts=node.characterizing_cuts_left,
                                                            cuts=cuts,
                                                            costs=costs,
                                                            verbose=verbose)
        unnormalized_p_right = compute_soft_predictions_node(characterizing_cuts=node.characterizing_cuts_right,
                                                             cuts=cuts,
                                                             costs=costs,
                                                             verbose=verbose)

        # normalize the ps
        total_p = unnormalized_p_left + unnormalized_p_right

        p_left = unnormalized_p_left / total_p
        p_right = unnormalized_p_right / total_p

        node.left_child.p = p_left * node.p
        node.right_child.p = p_right * node.p

        compute_soft_predictions_children(node=node.left_child,
                                          cuts=cuts,
                                          costs=costs,
                                          verbose=verbose)

        compute_soft_predictions_children(node=node.right_child,
                                          cuts=cuts,
                                          costs=costs,
                                          verbose=verbose)
