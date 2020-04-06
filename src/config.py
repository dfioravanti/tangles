from pathlib import Path
from argparse import Namespace

import yaml

# Global constants for validations of inputs

# Experiments
EXPERIMENT_SINGLE = 'single'
EXPERIMENT_BATCH = 'batch'

VALID_EXPERIMENTS = [
    EXPERIMENT_SINGLE,
    EXPERIMENT_BATCH
]

# Datasets

DATASET_QUESTIONNAIRE_SYNTHETIC = "q_syn"
DATASET_BINARY_IRIS = "iris"
DATASET_SBM = "sbm"
DATASET_KNN_BLOBS = "knn_blobs"
DATASET_MULTILEVEL = "multilevel"
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
    DATASET_KNN_BLOBS,
    DATASET_MULTILEVEL,
    DATASET_RING_OF_CLIQUES,
    DATASET_FLORENCE,
]

VALID_DATASETS = DISCRETE_DATASETS + GRAPH_DATASETS

# Preprocessing

PREPROCESSING_FEATURES = "fea"
PREPROCESSING_KMODES = "kmodes"
PREPROCESSING_KARNIG_LIN = "karnig_lin"

VALID_PREPROCESSING = [
    PREPROCESSING_FEATURES,
    PREPROCESSING_KARNIG_LIN,
    PREPROCESSING_KMODES
]

# Algorithm

ALGORITHM_CORE = "core"

VALID_ALGORITHM = [
    ALGORITHM_CORE
]


def load_validate_settings(root_dir):
    main_cfg_file = 'settings.yaml'
    main_cfg = load_settings(f'{root_dir}/{main_cfg_file}')
    experiment_type = main_cfg['experiment']['type']

    if experiment_type not in VALID_EXPERIMENTS:
        raise ValueError(f'The experiment type must be in: {VALID_EXPERIMENTS}')
    elif experiment_type == EXPERIMENT_SINGLE:
        auxiliary_cfg_file = 'settings_single.yaml'
    elif experiment_type == EXPERIMENT_BATCH:
        auxiliary_cfg_file = 'settings_batch.yaml'
    auxiliary_cfg = load_settings(f'{root_dir}/{auxiliary_cfg_file}')

    cfg = merge_config(main_cfg, auxiliary_cfg)
    args = dict_to_namespace(cfg)
    args = validate_settings(args)

    return args


def merge_config(main_cfg, auxiliary_cfg):
    """
    Updates the default configuration with the experiment specific one

    :param main_cfg:
    :param auxiliary_cfg:
    :return:
    """

    for k in auxiliary_cfg.keys():
        if k not in main_cfg.keys():
            main_cfg[k] = auxiliary_cfg[k]
        else:
            if isinstance(auxiliary_cfg[k], dict):
                assert isinstance(main_cfg[k], dict)
                merge_config(main_cfg[k], auxiliary_cfg[k])
            else:
                main_cfg[k] = auxiliary_cfg[k]

    return main_cfg


def dict_to_namespace(args):
    """
    Transforms the dictionary of the settings to a namespace
    """

    namespace = Namespace()

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
            if isinstance(getattr(namespace, attr), Namespace):
                result.update({attr: namespace_to_dict(getattr(namespace, attr))})
            else:
                result.update({attr: getattr(namespace, attr)})
    return result


def load_settings(file):
    with open(file, 'r') as f:
        return yaml.load(f, Loader=yaml.UnsafeLoader)


def validate_settings(args):

    if args.experiment.type not in VALID_EXPERIMENTS:
        raise ValueError(f'The experiment type must be in: {VALID_EXPERIMENTS}')

    if args.experiment.dataset_name not in VALID_DATASETS:
        raise ValueError(f'The dataset name must be in: {VALID_DATASETS}')

    if args.experiment.dataset_name in DISCRETE_DATASETS:
        args.experiment.dataset_type = 'discrete'
    elif args.experiment.dataset_name in GRAPH_DATASETS:
        args.experiment.dataset_type = 'graph'

    if args.preprocessing.name not in VALID_PREPROCESSING:
        raise ValueError(f'The preprocessing name must be in: {VALID_PREPROCESSING}')

    if args.algorithm.name not in VALID_ALGORITHM:
        raise ValueError(f'The algorithm name must be in: {VALID_ALGORITHM}')

    return args


def set_up_dirs(args, root_dir):
    args.output.dir = Path(f"{root_dir / args.output.dir}")
    args.output.dir.mkdir(parents=True, exist_ok=True)

    return args
