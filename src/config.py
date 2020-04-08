from pathlib import Path
from argparse import Namespace

import numpy as np

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
PREPROCESSING_FID_MAT = "fid_mat"
PREPROCESSING_COARSENING = "coarsening"

VALID_PREPROCESSING = [
    PREPROCESSING_FEATURES,
    PREPROCESSING_KARNIG_LIN,
    PREPROCESSING_KMODES,
    PREPROCESSING_COARSENING,
    PREPROCESSING_FID_MAT
]

# Algorithm

ALGORITHM_CORE = "core"

VALID_ALGORITHM = [
    ALGORITHM_CORE
]


def load_validate_settings(args_parser, root_dir):
    main_cfg_file = 'settings.yaml'

    main_cfg = load_settings(f'{root_dir}/{main_cfg_file}')

    cfg = merge_config(args_parser, main_cfg)
    args = dict_to_namespace(cfg)
    args = validate_settings(args)

    args.prefix = get_prefix(args)

    return args


def get_prefix(args):

    if args.experiment.dataset_name == DATASET_SBM:
        prefix = f'SMB_{len(args.dataset.sbm.block_sizes)}'
    elif args.experiment.dataset_name == DATASET_KNN_BLOBS:
        prefix = f'knn_blobs_{len(args.dataset.knn_blobs.blob_sizes)}'

    return prefix


def merge_config(args_parser, main_cfg):
    """
    Updates the default configuration with the experiment specific one

    :param main_cfg:
    :param auxiliary_cfg:
    :return:
    """

    if args_parser.seeds is not None:
        main_cfg['seeds'] = args_parser.seeds

    if args_parser.dataset_name == DATASET_SBM:
        if args_parser.sbm_bs is not None:
            main_cfg['dataset'][DATASET_SBM]['block_sizes'] = args_parser.sbm_bs
        if args_parser.sbm_ps is not None:
            main_cfg['dataset'][DATASET_SBM]['ps'] = args_parser.sbm_ps
        if args_parser.sbm_qs is not None:
            main_cfg['dataset'][DATASET_SBM]['qs'] = args_parser.sbm_qs
    elif args_parser.dataset_name == DATASET_KNN_BLOBS:
        if args_parser.gauss_bs is not None:
            main_cfg['dataset'][DATASET_KNN_BLOBS]['blob_sizes'] = args_parser.gauss_bs
        if args_parser.gauss_cs is not None:
            centers = np.array(args_parser.gauss_cs).reshape(2, -1).tolist()
            main_cfg['dataset'][DATASET_KNN_BLOBS]['blobs_centers'] = [centers]
        if args_parser.gauss_ks is not None:
            main_cfg['dataset'][DATASET_KNN_BLOBS]['ks'] = args_parser.gauss_ks

    if args_parser.pre_type == PREPROCESSING_KARNIG_LIN:
        if args_parser.KL_frac is not None:
            main_cfg['preprocessing']['karnig_lin']['nb_cuts'] = args_parser.KL_nb
        if args_parser.KL_frac is not None:
            main_cfg['preprocessing']['karnig_lin']['fractions'] = args_parser.KL_frac

    if args_parser.plot_tangles is not None:
        main_cfg['plot']['tangles'] = args_parser.plot_tangles
    if args_parser.plot_cuts is not None:
        main_cfg['plot']['cuts'] = args_parser.plot_cuts

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
        if attr[:1] != '_':
            if isinstance(getattr(namespace, attr), Namespace):
                result.update({attr: namespace_to_dict(getattr(namespace, attr))})
            else:
                result.update({attr: getattr(namespace, attr)})
    return result


def load_settings(file):
    with open(file, 'r') as f:
        return yaml.load(f, Loader=yaml.UnsafeLoader)


def validate_settings(args):

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
    args.root_dir = Path(f"{root_dir / 'output' / args.prefix}")
    args.plot_dir = Path(f"{args.root_dir / 'plots'}")

    args.root_dir.mkdir(parents=True, exist_ok=True)

    return args
