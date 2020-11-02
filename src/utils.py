import hashlib
import json

import numpy as np
from sklearn.manifold import TSNE


class Orientation(object):

    def __init__(self, direction):
        self.orientation_bool = direction
        if direction == 'both':
            self.direction = direction
        elif direction is True:
            self.direction = 'left'
        elif direction is False:
            self.direction = 'right'

    def __eq__(self, value):

        if self.direction == value.direction:
            return True

        return False

    def __str__(self):
        if self.direction == 'both':
            return self.direction
        elif self.direction == 'left':
            return 'True'
        elif self.direction == 'right':
            return 'False'


def get_hash(d):
    """
    Get the hash of a dictionary
    Parameters
    ----------
    d: dict
        The dictionary to hash

    Returns
    -------
    String
        The md5 hash of the dictionary
    """

    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode('utf-8')).hexdigest()


def normalize(array):
    """
    Normalize a 1d numpy array between [0,1]

    Parameters
    ----------
    array: ndarray
        the array to normalize

    Returns
    -------
    ndarray
        the normalized array
    """

    ptp = np.ptp(array)
    if ptp != 0:
        return (array - np.min(array)) / np.ptp(array)
    else:
        return np.ones_like(array)


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


def get_points_to_plot(xs, cs):
    """
    Calculate embedding of points for visualization

    Parameters
    ----------
    xs: ndarray
        the datapoints
    cs: ndarray
        the centers of the clusters if method supports

    Returns
    -------
    xs_embedded:
         embedding of points in two dimensions
    cs_embedded:
         embedding of centers if method supports
    """
    _, nb_features = xs.shape

    nb_centers = None
    cs_embedded = None
    if cs is not None:
        nb_centers, _ = cs.shape

    if nb_features > 2:
        if cs is not None:
            points_to_embed = np.vstack([xs, cs])
            embeds = TSNE(n_components=2, random_state=42).fit_transform(points_to_embed)
            xs_embedded, cs_embedded = embeds[:-nb_centers], embeds[-nb_centers:]
        else:
            xs_embedded = TSNE(n_components=2, random_state=42).fit_transform(xs)
    else:
        xs_embedded = xs
        cs_embedded = cs

    if cs is not None:
        return xs_embedded, cs_embedded
    else:
        return xs_embedded, None

# def dict_product(dicts):
#     """
#      list(dict_product(dict(number=[1,2], character='ab')))
#     [{'character': 'a', 'number': 1},
#      {'character': 'a', 'number': 2},
#      {'character': 'b', 'number': 1},
#      {'character': 'b', 'number': 2}]
#     """
#     return (dict(zip(dicts.keys(), x)) for x in itertools.product(*dicts.values()))


# def change_lower(interval, new_value):
#     if new_value > interval[0]:
#         return new_value, interval[1]
#
#     return interval
#
#
# def change_upper(interval, new_value):
#     if new_value < interval[1]:
#         return interval[0], new_value
#
#     return interval

def subset(a, b):
    return (a & b).count() == a.count()
