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
DATASET_POLITICAL_BOOKS = "pol_books"
DATASET_FLORENCE = "flo"
DATASET_BIG5 = 'big5'
DATASET_CANCER = 'cancer'

DISCRETE_DATASETS = [
    DATASET_QUESTIONNAIRE_SYNTHETIC,
    DATASET_BINARY_IRIS,
    DATASET_BIG5,
    DATASET_CANCER
]

GRAPH_DATASETS = [
    DATASET_SBM,
    DATASET_KNN_BLOBS,
    DATASET_MULTILEVEL,
    DATASET_POLITICAL_BOOKS,
    DATASET_FLORENCE,
]

VALID_DATASETS = DISCRETE_DATASETS + GRAPH_DATASETS

# Preprocessing

PREPROCESSING_USE_FEATURES = "features"
PREPROCESSING_KMODES = "kmodes"
PREPROCESSING_KARNIG_LIN = "karnig_lin"
PREPROCESSING_FID_MAT = "fid_mat"
PREPROCESSING_COARSENING = "coarsening"

VALID_PREPROCESSING = [
    PREPROCESSING_USE_FEATURES,
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

NAN = -9999

def load_validate_settings(args_parser, root_dir):
    main_cfg_file = 'settings.yaml'

    main_cfg = load_settings(f'{root_dir}/{main_cfg_file}')

    args = merge_config(args_parser, main_cfg)
    args = validate_settings(args)

    args = delete_useless_parameters(args)

    args['prefix'] = get_prefix(args)

    return args


def delete_useless_parameters(args):

    dataset_name = args['experiment']['dataset_name']
    value = args['dataset'][dataset_name]
    args['dataset'].clear()
    args['dataset'] = value

    preprocessing_name = args['experiment']['preprocessing_name']
    if preprocessing_name != PREPROCESSING_USE_FEATURES:
        value = args['preprocessing'][preprocessing_name]
        args['preprocessing'].clear()
        args['preprocessing'] = value
    else:
        args['preprocessing'].clear()
    return args


def get_prefix(args):

    if args['experiment']['dataset_name'] == DATASET_SBM:
        prefix = f'SMB_{len(args["dataset"]["block_sizes"])}'
    elif args['experiment']['dataset_name'] == DATASET_KNN_BLOBS:
        prefix = f'knn_blobs_{len(args["dataset"]["blob_sizes"])}'
    else:
        prefix = args['experiment']['dataset_name']

    return prefix


def merge_config(args_parser, main_cfg):
    """
    Updates the default configuration with the experiment specific one

    :param main_cfg:
    :param auxiliary_cfg:
    :return:
    """

    if args_parser.seed is not None:
        main_cfg['seed'] = args_parser.seed

    if args_parser.dataset_name is not None:
        main_cfg['experiment']['dataset_name'] = args_parser.dataset_name
    if args_parser.preprocessing_name is not None:
        main_cfg['experiment']['preprocessing_name'] = args_parser.preprocessing_name
    if args_parser.agreement is not None:
        main_cfg['experiment']['agreement'] = args_parser.agreement
    if args_parser.percentile_orders is not None:
        main_cfg['experiment']['percentile_orders'] = args_parser.percentile_orders

    if args_parser.dataset_name == DATASET_SBM:
        if args_parser.sbm_bs is not None:
            main_cfg['dataset'][DATASET_SBM]['block_sizes'] = args_parser.sbm_bs
        if args_parser.sbm_p is not None:
            main_cfg['dataset'][DATASET_SBM]['p'] = args_parser.sbm_p
        if args_parser.sbm_q is not None:
            main_cfg['dataset'][DATASET_SBM]['q'] = args_parser.sbm_q
    elif args_parser.dataset_name == DATASET_KNN_BLOBS:
        if args_parser.gauss_bs is not None:
            main_cfg['dataset'][DATASET_KNN_BLOBS]['blob_sizes'] = args_parser.gauss_bs
        if args_parser.gauss_cs is not None:
            centers = np.array(args_parser.gauss_cs).reshape(2, -1).tolist()
            main_cfg['dataset'][DATASET_KNN_BLOBS]['blob_centers'] = centers
        if args_parser.gauss_k is not None:
            main_cfg['dataset'][DATASET_KNN_BLOBS]['k'] = args_parser.gauss_k

    if args_parser.preprocessing_name == PREPROCESSING_KARNIG_LIN:
        if args_parser.nb_cuts is not None:
            main_cfg['preprocessing']['karnig_lin']['nb_cuts'] = args_parser.nb_cuts
        if args_parser.lb_f is not None:
            main_cfg['preprocessing']['karnig_lin']['lb_f'] = args_parser.lb_f
    elif args_parser.preprocessing_name == PREPROCESSING_FID_MAT:
        if args_parser.nb_cuts is not None:
            main_cfg['preprocessing']['fid_mat']['nb_cuts'] = args_parser.nb_cuts
        if args_parser.lb_f is not None:
            main_cfg['preprocessing']['fid_mat']['lb_f'] = args_parser.lb_f

    if args_parser.plot_tangles is not None:
        main_cfg['plot']['tangles'] = args_parser.plot_tangles
    if args_parser.plot_cuts is not None:
        main_cfg['plot']['cuts'] = args_parser.plot_cuts

    if args_parser.unique_id is not None:
        main_cfg['experiment']['unique_id'] = str(args_parser.unique_id)

    return main_cfg


def load_settings(file):
    with open(file, 'r') as f:
        return yaml.load(f, Loader=yaml.UnsafeLoader)


def validate_settings(args):

    if args['experiment']['dataset_name'] not in VALID_DATASETS:
        raise ValueError(f'The dataset name must be in: {VALID_DATASETS}')

    if args['experiment']['dataset_name'] in DISCRETE_DATASETS:
        args['experiment']['dataset_type'] = 'discrete'
    elif args['experiment']['dataset_name'] in GRAPH_DATASETS:
        args['experiment']['dataset_type'] = 'graph'

    if args['experiment']['preprocessing_name'] not in VALID_PREPROCESSING:
        raise ValueError(f'The preprocessing name must be in: {VALID_PREPROCESSING}')

    return args


def set_up_dirs(args, root_dir):
    args['root_dir'] = Path(f"{root_dir / 'output' / args['experiment']['unique_id'] / args['prefix']}")
    args['plot_dir'] = Path(f"{args['root_dir'] / 'plots'}")

    args['plot_dir'].mkdir(parents=True, exist_ok=True)

    return args
