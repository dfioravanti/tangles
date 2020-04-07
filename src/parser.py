import argparse
from types import SimpleNamespace


def make_parser():
    """
    Build the command line parser for the program.

    Returns
    -------
    parser: ArgumentParser
        The parser

    """

    parser = argparse.ArgumentParser(description='Program to compute tangles')

    parser.add_argument('-t', dest='dataset_name', action='store')
    parser.add_argument('-p', dest='pre_type', action='store')
    parser.add_argument('-s', dest='seeds', nargs='+', type=int)

    # SBM
    parser.add_argument('--sbm_bs', dest='sbm_bs', nargs='+', type=int)
    parser.add_argument('--sbm_ps', dest='sbm_ps', nargs='+', type=float)
    parser.add_argument('--sbm_qs', dest='sbm_qs', nargs='+', type=float)

    # KL algorithm
    parser.add_argument('--KL_nb', dest='KL_nb', action='store', type=int)
    parser.add_argument('--KL_frac', dest='KL_frac', nargs='+', type=int)

    # Plotting
    parser.add_argument('--plot_tangles', dest='plot_tangles', action='store_true')
    parser.add_argument('--no_plot_tangles', dest='plot_tangles', action='store_false')

    parser.add_argument('--plot_cuts', dest='plot_cuts', action='store_true')
    parser.add_argument('--no_plot_cuts', dest='plot_cuts', action='store_false')

    return parser


def validate_args(parser):
    """
    TODO: To this function

    Given a parser we validate the parameters of the program

    Parameters
    ----------
     parser: ArgumentParser
        The parser

    Returns
    -------
    args: SimpleNamespace
        The parameters of the program parsed and validated into a SimpleNamespace
    """

    pass


def get(namespace, key):
    """
    Returns the value associated to key in the namespace. If key does not exists returns None.

    Parameters
    ----------
    namespace: SimpleNamespace
        The namespace we are examining for the key
    key: String
        The key we are interested in

    Returns
    -------
    value: Object
        The value associated with the key. None if not present
    """

    value = namespace.__dict__.get(key)
    return value


def add_value(namespace, key, value):
    """
    Add a couple (key, value) to a namespace.
    We assume that that the key is in the form k1.k2...kn where only the last node has a value

    Parameters
    ----------
    namespace: SimpleNamespace
        The namespace we want to add a value to
    key: String
        The key associated with the value
    value: Object
        The value to add to the namespace

    Returns
    -------
    namespace: SimpleNamespace
        The namespace with the added (key, value)

    """

    key_str = key.split('.')
    current_key = key_str[0]
    remaning_keys = '.'.join(key_str[1:])
    current_value = get(namespace, current_key)

    if len(remaning_keys) == 0:
        new_value = value
    else:
        if current_value is None:
            new_value = add_value(SimpleNamespace(), remaning_keys, value)
        else:
            new_value = add_value(current_value, remaning_keys, value)

    setattr(namespace, current_key, new_value)
    return namespace


def to_SimpleNamespace(args):
    """
    Transforms a argparse.Namespace into a SimpleNamespace.
    We assume that that all the keys are in the form k1.k2...kn where only the last node has a value

    Parameters
    ----------
    args: argparse.Namespace
        The namespace we want to convert

    Returns
    -------
    namespace: SimpleNamespace
        The converted namespace
    """

    namespace = SimpleNamespace()

    for key, value in vars(args).items():
        namespace = add_value(namespace, key, value)

    return namespace