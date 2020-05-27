import itertools
import hashlib
import json

import numpy as np
from numpy import pi, cos, sin, sqrt, arange
from sklearn.manifold import TSNE

class Orientation(object):

    def __init__(self, direction):
        if direction == 'both':
            self.direction = direction
        elif direction == True:
            self.direction = 'left'
        elif direction == False:
            self.direction = 'right'

    def __eq__(self, value):
        
        if self.direction == 'both' or value.direction == 'both':
            return True
        if self.direction == value.direction:
            return True

        return False
    
    def __str__(self):
        if self.direction == 'both':
            return direction
        elif self.direction == 'left':
            return 'True'
        elif self.direction == 'right':
            return 'False'


def get_id(d):
    return hashlib.md5(json.dumps(d, sort_keys=True).encode('utf-8')).hexdigest()

def normalize(array):
    ptp = np.ptp(array)
    if ptp != 0:
        return (array - np.min(array))/np.ptp(array)
    else:
        return array


def matching_items(d1, d2):

    matching_keys = []
    common_keys = d1.keys() & d2.keys()
    for k in common_keys:
        if d1[k] == d2[k]:
            matching_keys.append(k)

    return matching_keys


def merge_dictionaries_with_disagreements(d1, d2):

    merge = {**d1, **d2}
    common_keys = d1.keys() & d2.keys()

    for k in common_keys:
        if d1[k] != d2[k]:
            merge.pop(k)

    return merge



def get_positions_from_labels(ys):
    positions = np.zeros([len(ys), 2])
    classes = np.unique(ys)

    num_pts = len(classes)
    indices = np.arange(0, num_pts, dtype=float) + 0.5

    r = np.sqrt(indices / num_pts)
    theta = np.pi * (1 + 5 ** 0.5) * indices

    means = np.transpose([r * np.cos(theta), r * np.sin(theta)]) * 20

    for i, c in enumerate(classes):
        number = sum(ys == c)
        pos = np.random.normal(means[i], [2, 2], [number, 2])
        positions[ys == c, :] = pos

    return positions


def get_points_to_plot(xs, cs):
    _, nb_features = xs.shape
    if cs is not None:
        nb_centers, _  = cs.shape

    if nb_features > 2:
        if cs is not None:
            points_to_embed = np.vstack([xs, cs])
            embeds = TSNE(n_components=2, metric='manhattan', random_state=42).fit_transform(points_to_embed)
            xs_embedded, cs_embedded = embeds[:-nb_centers], embeds[-nb_centers:]
        else:
            xs_embedded = TSNE(n_components=2, metric='manhattan', random_state=42).fit_transform(xs)
    else:
        xs_embedded = xs

    if cs is not None:
        return xs_embedded, cs_embedded
    else:
        return xs_embedded, None


def dict_product(dicts):

    """
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts.keys(), x)) for x in itertools.product(*dicts.values()))


def change_lower(interval, new_value):
    if new_value > interval[0]:
        return new_value, interval[1]

    return interval


def change_upper(interval, new_value):
    if new_value < interval[1]:
        return interval[0], new_value

    return interval
