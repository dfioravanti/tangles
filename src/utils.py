import itertools

import numpy as np

from sklearn.manifold import TSNE

def get_points_to_plot(xs, cs):
    _, nb_features = xs.shape
    if cs is not None:
        nb_centers, _  = cs.shape

    if nb_features > 2:
        if cs is not None:
            points_to_embed = np.vstack([xs, cs])
            embeds = TSNE(n_components=2, metric='manhattan', perplexity=10).fit_transform(points_to_embed)
            xs_embedded, cs_embedded = embeds[:-nb_centers], embeds[-nb_centers:]
        else:
            xs_embedded = TSNE(n_components=2, metric='manhattan').fit_transform(xs)
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
