from copy import deepcopy

import matplotlib.pyplot as plt

import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout, write_dot

import bitarray as ba
import numpy as np

from src.tangles import Tangle
from src.utils import matching_items, Orientation
        

class TangleNode(object):

    def __init__(self, parent, right_child, left_child, is_left_child, \
                       did_split, last_cut_added_id, last_cut_added_orientation, tangle):

        self.parent = parent
        self.right_child = right_child
        self.left_child = left_child
        self.is_left_child = is_left_child
        
        self.did_split = did_split
        self.last_cut_added_id = last_cut_added_id
        self.last_cut_added_orientation = last_cut_added_orientation
        
        self.tangle = tangle
        
    def __str__(self, height=0):
        
        if self.parent is None:
            string = 'Root'
        else:
            padding = ' '
            string = f'{padding*height}{self.last_cut_added_id} -> {self.last_cut_added_orientation}'
            
        if self.left_child is not None:
            string += '\n'
            string += self.left_child.__str__(height=height+1)
        if self.right_child is not None:
            string += '\n'
            string += self.right_child.__str__(height=height+1)
            
        return string
    

class ContractedTangleNode(TangleNode):

    def __init__(self, parent, node):

        attributes = node.__dict__
        super().__init__(**attributes)

        self.parent = parent
        self.right_child = self.right_child = None

        self.characterizing_cuts = dict()
        self.characterizing_cuts_left = dict()
        self.characterizing_cuts_right = dict()

        self.is_left_child_processed = False
        self.is_right_child_processed = False       

        self.p = None 
        
    def __str__(self, height=0):
        
        if self.parent is None:
            string = 'Root'
        else:
            padding = ' '
            string_cuts = [f'{k} -> {v}' for k, v in self.characterizing_cuts.items()]
            string = f'{padding*height}- {string_cuts}'
                    
        if self.left_child is not None:
            string += '\n'
            string += self.left_child.__str__(height=height+1)
        if self.right_child is not None:
            string += '\n'
            string += self.right_child.__str__(height=height+1)
            
        return string

class TangleTree(object):
    
    def __init__(self):
        
        self.root = TangleNode(parent=None,
                               right_child=None, 
                               left_child=None, 
                               is_left_child=None, 
                               did_split=None,
                               last_cut_added_id=None,
                               last_cut_added_orientation=None, 
                               tangle=None)
        self.active = []
        self.maximals = []
        self.will_split = []
        self.is_empty = True
        
    def __str__(self):
        return str(self.root)

    def add_cut(self, cut, cut_id, agreement):
        
        if self.is_empty:

            could_add_one, did_split, is_maximal = self._add_cut_to_empty_tree(cut=cut, cut_id=cut_id, agreement=agreement)

            if is_maximal:
                self.maximals.append(self.root)
            else:
                self.is_empty = False

            if did_split:
                self.will_split.append(self.root)
            
        else:

            current_active = self.active
            self.active = []

            could_add_one_to_active = False
            for current_node in current_active:
                could_add_one, did_split, is_maximal = self._add_cut_to_node(current_node, cut, cut_id, agreement)
                could_add_one = could_add_one or could_add_one_to_active
                
                if did_split:
                    self.will_split.append(current_node)
                elif is_maximal:
                    self.maximals.append(current_node)

        return could_add_one

    def _add_cut_to_node(self, current_node, cut, cut_id, agreement):

        old_tangle = current_node.tangle
        new_tangle_true = old_tangle.add(new_cut=ba.bitarray(cut.tolist()),
                                         new_cut_specification={cut_id: True},
                                         min_size=agreement)
        new_tangle_false = old_tangle.add(new_cut=ba.bitarray((~cut).tolist()),
                                          new_cut_specification={cut_id: False},
                                          min_size=agreement)

        could_add_one, did_split, is_maximal = self._add_children(current_node=current_node,
                                                                  new_tangle_true=new_tangle_true,
                                                                  new_tangle_false=new_tangle_false,
                                                                  cut_id=cut_id)

        return could_add_one, did_split, is_maximal
        
    def _add_cut_to_empty_tree(self, cut, cut_id, agreement):
        
        if np.sum(cut) >= agreement:
            array = ba.bitarray(cut.tolist())
            new_tangle_true = Tangle(cuts=[array],
                                     core=[array],
                                     specification={cut_id: True})
        else:
            new_tangle_true = None

        if np.sum(~cut) >= agreement:
            array = ba.bitarray((~cut).tolist())
            new_tangle_false = Tangle(cuts=[array],
                                     core=[array],
                                     specification={cut_id: False})
        else:
            new_tangle_false = None

        could_add_one, did_split, is_maximal = self._add_children(current_node=self.root,
                                                                  new_tangle_true=new_tangle_true,
                                                                  new_tangle_false=new_tangle_false,
                                                                  cut_id=cut_id)

        return could_add_one, did_split, is_maximal

    def _add_children(self, current_node, new_tangle_true, new_tangle_false, cut_id):

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
            new_node = self._add_child(current_node=current_node,
                                       tangle=new_tangle_true, 
                                       last_cut_added_id=cut_id, 
                                       last_cut_added_orientation=True, 
                                       did_split=did_split)
            self.active.append(new_node)

        if new_tangle_false is not None:
            could_add_one = True
            new_node = self._add_child(current_node=current_node,
                                       tangle=new_tangle_false, 
                                       last_cut_added_id=cut_id, 
                                       last_cut_added_orientation=False, 
                                       did_split=did_split)
            self.active.append(new_node)

        return could_add_one, did_split, is_maximal

    def _add_child(self, current_node, tangle, last_cut_added_id, last_cut_added_orientation, did_split):

        new_node = TangleNode(parent=current_node,
                   right_child=None, 
                   left_child=None, 
                   is_left_child=last_cut_added_orientation, 
                   did_split=did_split,
                   last_cut_added_id=last_cut_added_id,
                   last_cut_added_orientation=last_cut_added_orientation, 
                   tangle=tangle)

        if new_node.is_left_child:
            current_node.left_child = new_node
        else:
            current_node.right_child = new_node

        return new_node

    def plot_tree(self, node=None, depth=0, position=0, set_header=True, path=None):

        tree = nx.Graph()
        labels = self._add_node_to_nx(tree, self.root)

        pos = graphviz_layout(tree, prog='dot')

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
        nx.draw_networkx(tree, pos=pos, ax=ax, labels=labels, node_size = 1500)
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
        
        self.maximals = []
        self.will_split = []
        self.root = self._convert_subtree(parent=None, node=tree.root)

    def __str__(self):
        return str(self.root)

    def _convert_subtree(self, parent, node):

        contracted_node = ContractedTangleNode(parent=parent, node=node)
        if node.left_child is not None:
            contracted_left_child = self._convert_subtree(parent=contracted_node, node=node.left_child)
            contracted_node.left_child = contracted_left_child
        if node.right_child is not None:
            contracted_right_child = self._convert_subtree(parent=contracted_node, node=node.right_child)
            contracted_node.right_child = contracted_right_child

        if node.left_child is None and node.right_child is None:
            self.maximals.append(contracted_node)
        if node.left_child is not None and node.right_child is not None:
            self.will_split.append(contracted_node)

        return contracted_node


def contract_until_split(node):

    node.characterizing_cuts[node.last_cut_added_id] = Orientation(node.last_cut_added_orientation)
    while not node.did_split and node.parent.parent is not None:
        
        parent = node.parent
        node.characterizing_cuts[parent.last_cut_added_id] = Orientation(parent.last_cut_added_orientation)

        if parent.is_left_child:
            parent.parent.left_child = node
        else:
            parent.parent.right_child = node
        
        node.parent = parent.parent
        node.is_left_child = parent.is_left_child
        node.did_split = parent.did_split

def process_split(node):

    characterizing_cuts_left = node.left_child.characterizing_cuts
    characterizing_cuts_right = node.right_child.characterizing_cuts

    id_cuts_oriented_same_way = matching_items(characterizing_cuts_left, characterizing_cuts_right)
    for id_cut in id_cuts_oriented_same_way:
        characterizing_cuts_left.pop(id_cut)
        characterizing_cuts_right.pop(id_cut)

    id_cuts_oriented_both_ways = characterizing_cuts_left.keys() & characterizing_cuts_right.keys()
    characterizing_cuts = {**characterizing_cuts_left, **characterizing_cuts_right}
    for id_cut in id_cuts_oriented_both_ways:
        characterizing_cuts[id_cut] = Orientation('both')

    node.characterizing_cuts_left = characterizing_cuts_left
    node.characterizing_cuts_right = characterizing_cuts_right
    node.characterizing_cuts = characterizing_cuts


def contract_tree(tree):

    contracted_tree = ContractedTangleTree(tree)

    will_split = []
    nodes_to_contract = []

    for node in contracted_tree.maximals:
        nodes_to_contract.append(node)

    done = False

    while not done:

        # Contract subtrees

        for node in nodes_to_contract:
            contract_until_split(node)
            
            if node.is_left_child:
                will_split.append(node.parent)
                node.parent.is_left_child_processed = True
            else:
                node.parent.is_right_child_processed = True

            if node.parent.parent is None and not node.did_split:
                done = True

        nodes_to_contract = []

        # The next cut will splitting

        i_to_pop = []
        for i, node in enumerate(will_split):
            if node.is_left_child_processed and node.is_right_child_processed:
                
                if node.parent is None:
                    done = True

                i_to_pop.append(i)
                process_split(node)
                nodes_to_contract.append(node)            
        
        for i in i_to_pop[::-1]:
            will_split.pop(i)


    return contracted_tree


def compute_soft_predictions_node(node, characterizing_cuts, cuts, orders, cost_function, verbose):

    if node.left_child is None and node.right_child is None:
        if not node.last_cut_added_id in characterizing_cuts:
            characterizing_cuts[node.last_cut_added_id] = Orientation(node.last_cut_added_orientation)

    idx_characterizing_cuts, orientation_characterizing_cuts = [], []
    for i, o in characterizing_cuts.items():
        if o.direction == 'left':
            idx_characterizing_cuts.append(i)
            orientation_characterizing_cuts.append(True)
        elif o.direction == 'right':
            idx_characterizing_cuts.append(i)
            orientation_characterizing_cuts.append(False)

    costs = cost_function(orders[idx_characterizing_cuts])

    p = np.sum((cuts[idx_characterizing_cuts, :].T == orientation_characterizing_cuts) * costs, axis=1)

    return p


def compute_soft_predictions_children(node, cuts, orders, cost_function, verbose):

    _, nb_points = cuts.shape
    
    if node.parent is None:
        node.p = np.ones(nb_points)  
    
    if node.left_child is not None and node.right_child is None:
        node.left_child.p = np.ones(nb_points)
        compute_soft_predictions_children(node=node.left_child, 
                                          cuts=cuts,
                                          orders=orders, 
                                          cost_function=cost_function,
                                          verbose=verbose)    
    elif node.left_child is None and node.right_child is not None:
        node.right_child.p = np.ones(nb_points)
        compute_soft_predictions_children(node=node.right_child, 
                                          cuts=cuts,
                                          orders=orders, 
                                          cost_function=cost_function,
                                          verbose=verbose)
    elif node.left_child is not None and node.right_child is not None:
        p_left = np.zeros(nb_points)
        p_right = np.zeros(nb_points)

        unnormalized_p_left = compute_soft_predictions_node(node=node.left_child, 
                                                            characterizing_cuts=node.characterizing_cuts_left,
                                                            cuts=cuts, 
                                                            orders=orders, 
                                                            cost_function=cost_function,
                                                            verbose=verbose)
        unnormalized_p_right = compute_soft_predictions_node(node=node.right_child, 
                                                            characterizing_cuts=node.characterizing_cuts_right,
                                                            cuts=cuts, 
                                                            orders=orders, 
                                                            cost_function=cost_function,
                                                            verbose=verbose)

        total_p = unnormalized_p_left + unnormalized_p_right
            
        p_left = unnormalized_p_left / total_p
        p_right = unnormalized_p_right / total_p

        node.left_child.p = p_left * node.p
        node.right_child.p = p_right * node.p

        compute_soft_predictions_children(node=node.left_child, 
                                        cuts=cuts,
                                        orders=orders, 
                                        cost_function=cost_function,
                                        verbose=verbose)   

        compute_soft_predictions_children(node=node.right_child, 
                                        cuts=cuts,
                                        orders=orders, 
                                        cost_function=cost_function,
                                        verbose=verbose)

def compute_hard_predictions_node(node, idx_points, max_tangles):

    if node.left_child is None and node.right_child is None:

        y = max_tangles.index(node)
        return {y: idx_points}

    if node.right_child is None:

        left_cluster = compute_hard_predictions_node(node.left_child, idx_points, max_tangles)
        right_cluster = {}

    elif node.left_child is None:

        right_cluster = compute_hard_predictions_node(node.right_child, idx_points, max_tangles)
        left_cluster = {}

    else:
        
        p_left = node.left_child.p[idx_points]
        p_right = node.right_child.p[idx_points]

        idx_left = p_left >= p_right
        idx_right = ~idx_left

        left_cluster = compute_hard_predictions_node(node.left_child, idx_points[idx_left], max_tangles)
        right_cluster = compute_hard_predictions_node(node.right_child, idx_points[idx_right], max_tangles)

    return {**left_cluster, **right_cluster}
