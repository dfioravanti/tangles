import argparse
from types import SimpleNamespace

# Global constants for validations of inputs
# Datasets

DATASET_QUESTIONNAIRE_SYNTHETIC = "q_syn"
DATASET_BINARY_IRIS = "iris"
DATASET_SBM = "sbm"
DATASET_LFR = "lfr"
DATASET_RING_OF_CLIQUES = "roc"
DATASET_FLORENCE = "flo"
DATASET_BIG5 = 'big5'

VALID_DATASETS = [
    DATASET_QUESTIONNAIRE_SYNTHETIC,
    DATASET_BINARY_IRIS,
    DATASET_SBM,
    DATASET_LFR,
    DATASET_RING_OF_CLIQUES,
    DATASET_FLORENCE,
    DATASET_BIG5
]

# Preprocessing

PREPROCESSING_FEATURES = "fea"
PREPROCESSING_MAKE_SUBMODULAR = "sub"
PREPROCESSING_NEIGHBOURHOOD_CUTS = "nei"

VALID_PREPROCESSING = [
    PREPROCESSING_FEATURES,
    PREPROCESSING_MAKE_SUBMODULAR,
    PREPROCESSING_NEIGHBOURHOOD_CUTS
]

# Algorithm

ALGORITHM_CORE = "core"

VALID_ALGORITHM = [
    ALGORITHM_CORE
]


def make_parser():
    """
    Build the command line parser for the program.

    Returns
    -------
    parser: ArgumentParser
        The parser

    """

    parser = argparse.ArgumentParser(description='Program to compute tangles')

    # Datasets
    parser.add_argument('-p', '--dat_path', dest='dataset.path', action='store')
    parser.add_argument('-t', '--dat_type', dest='dataset.type', action='store')

    # Preprocessing
    parser.add_argument('--pre_name', dest='preprocessing.name', action='store', default='no')

    # Algorithm
    parser.add_argument('--alg_name', dest='algorithm.name', action='store')

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


def validate_settings(args):
    if args.preprocessing.name not in VALID_PREPROCESSING:
        raise ValueError(f'The preprocessing name must be in: {VALID_PREPROCESSING}')

    if args.algorithm.name not in VALID_ALGORITHM:
        raise ValueError(f'The algorithm name must be in: {VALID_ALGORITHM}')
