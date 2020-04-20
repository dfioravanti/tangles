import itertools


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
