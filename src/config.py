from pathlib import Path
from types import SimpleNamespace

import yaml

# Global constants for validations of inputs
# Datasets

DATASET_QUESTIONNAIRE_SYNTHETIC = "q_syn"
DATASET_BINARY_IRIS = "iris"
DATASET_SBM = "sbm"
DATASET_LFR = "lfr"
DATASET_RING_OF_CLIQUES = "roc"
DATASET_FLORENCE = "flo"
DATASET_BIG5 = 'big5'

DISCRETE_DATASETS = [
    DATASET_QUESTIONNAIRE_SYNTHETIC,
    DATASET_BINARY_IRIS,
    DATASET_BIG5
]

GRAPH_DATASETS = [
    DATASET_SBM,
    DATASET_LFR,
    DATASET_RING_OF_CLIQUES,
    DATASET_FLORENCE,
]

VALID_DATASETS = DISCRETE_DATASETS + GRAPH_DATASETS

# Preprocessing

PREPROCESSING_FEATURES = "fea"
PREPROCESSING_MAKE_SUBMODULAR = "sub"
PREPROCESSING_RANDOM_COVER = "random"
PREPROCESSING_KARGER = "karger"
PREPROCESSING_FAST_MINCUT = "fast"
PREPROCESSING_KMODES = "kmodes"

VALID_PREPROCESSING = [
    PREPROCESSING_FEATURES,
    PREPROCESSING_MAKE_SUBMODULAR,
    PREPROCESSING_RANDOM_COVER,
    PREPROCESSING_KARGER,
    PREPROCESSING_FAST_MINCUT,
    PREPROCESSING_KMODES
]

# Algorithm

ALGORITHM_CORE = "core"

VALID_ALGORITHM = [
    ALGORITHM_CORE
]


def load_validate_settings(root_dir):
    cfg_file = 'settings.yaml'

    cfg = load_settings(f'{root_dir}/{cfg_file}')
    args = dict_to_namespace(cfg)

    args = validate_settings(args)

    return args


def dict_to_namespace(args):
    """
    Transforms the dictionary of the settings to a namespace
    """

    namespace = SimpleNamespace()

    for key, value in args.items():
        keep = '~dict~' in key
        if keep:
            pos = key.find('~dict~')
            key = key[:pos] + key[pos + len('~dict~'):]

        if isinstance(value, dict) and not keep:
            value = dict_to_namespace(value)

        key = key.replace(' ', '_')
        setattr(namespace, key, value)

    return namespace


def namespace_to_dict(namespace):
    """
    Transforms a namespace into a dictionary
    """

    result = {}
    for attr in dir(namespace):
        if attr[:2] != '__':
            if isinstance(getattr(namespace, attr), SimpleNamespace):
                result.update({attr: namespace_to_dict(getattr(namespace, attr))})
            else:
                result.update({attr: getattr(namespace, attr)})
    return result


def load_settings(file):
    with open(file, 'r') as f:
        return yaml.load(f, Loader=yaml.UnsafeLoader)


def validate_settings(args):
    if args.dataset.name not in VALID_DATASETS:
        raise ValueError(f'The dataset name must be in: {VALID_DATASETS}')

    if args.dataset.name in DISCRETE_DATASETS:
        args.dataset.type = 'discrete'
    elif args.dataset.name in GRAPH_DATASETS:
        args.dataset.type = 'graph'

    if args.preprocessing.name not in VALID_PREPROCESSING:
        raise ValueError(f'The preprocessing name must be in: {VALID_PREPROCESSING}')

    if args.algorithm.name not in VALID_ALGORITHM:
        raise ValueError(f'The algorithm name must be in: {VALID_ALGORITHM}')

    return args


def set_up_dirs(args, root_dir):
    args.output.root_dir = Path(f"{root_dir / args.output.root_dir}")
    args.output.root_dir.mkdir(parents=True, exist_ok=True)

    return args
