from pathlib import Path

import numpy as np
import yaml

from src.types import Dataset, Preprocessing, CutFinding, CostFunction

NAN = -9999


def load_validate_parser(args):
    pass


def delete_useless_parameters(args):

    for operation in ['dataset', 'preprocessing', 'cut_finding', 'cost_function']:

        name = args['experiment'][operation].value
        try:
            value = args[operation][name]
        except KeyError:
            value = {}
        args[operation].clear()
        args[operation] = value

    return args


def validate_settings(args, mode='cfg_file'):

    args = validate_names(args)
    args['experiment']['unique_id'] = str(args['experiment']['unique_id'])

    if mode == 'cfg_file':
        args = delete_useless_parameters(args)

    return args


def load_validate_config_file(cfg_file_path):

    args = load_settings(cfg_file_path)
    args = validate_settings(args, mode='cfg_file')

    return args


def deactivate_plots(args):
    if args['plot']['no_plots']:
        for key in args['plot']:
            if key != 'no_plots':
                args['plot'][key] = False

    return args


def merge_config(args_parser, main_cfg):
    """
    Updates the default configuration with the experiment specific one

    :param main_cfg:
    :param auxiliary_cfg:
    :return:
    """

    if args_parser.seed is not None:
        main_cfg['experiment']['seed'] = args_parser.seed

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
    elif args_parser.dataset_name == DATASET_BLOBS:
        if args_parser.gauss_bs is not None:
            main_cfg['dataset'][DATASET_BLOBS]['blob_sizes'] = args_parser.gauss_bs
        if args_parser.gauss_cs is not None:
            centers = np.array(args_parser.gauss_cs).reshape(2, -1).tolist()
            main_cfg['dataset'][DATASET_BLOBS]['blob_centers'] = centers
        if args_parser.gauss_k is not None:
            main_cfg['dataset'][DATASET_BLOBS]['radius'] = args_parser.gauss_radius
    elif args_parser.dataset_name == DATASET_MINDSETS:
        if args_parser.mind_sizes is not None:
            main_cfg['dataset'][DATASET_MINDSETS]['mindset_sizes'] = args_parser.mind_sizes
        if args_parser.mind_questions is not None:
            main_cfg['dataset'][DATASET_MINDSETS]['nb_questions'] = args_parser.mind_questions
        if args_parser.mind_useless is not None:
            main_cfg['dataset'][DATASET_MINDSETS]['nb_useless'] = args_parser.mind_useless
        if args_parser.mind_noise is not None:
            main_cfg['dataset'][DATASET_MINDSETS]['noise'] = args_parser.mind_noise
    elif args_parser.dataset_name == DATASET_QUESTIONNAIRE:
        if args_parser.q_nb_samples is not None:
            main_cfg['dataset'][DATASET_QUESTIONNAIRE]['nb_samples'] = args_parser.q_nb_samples
        if args_parser.q_nb_features is not None:
            main_cfg['dataset'][DATASET_QUESTIONNAIRE]['nb_features'] = args_parser.q_nb_features
        if args_parser.q_nb_mindsets is not None:
            main_cfg['dataset'][DATASET_QUESTIONNAIRE]['nb_mindsets'] = args_parser.q_nb_mindsets
        if args_parser.q_range_answers is not None:
            main_cfg['dataset'][DATASET_QUESTIONNAIRE]['range_answers'] = args_parser.q_range_answers
            main_cfg['preprocessing'][PREPROCESSING_BINARIZED_LIKERT]['range_answers'] = args_parser.q_range_answers

    if args_parser.preprocessing_name == PREPROCESSING_KARNIG_LIN:
        if args_parser.nb_cuts is not None:
            main_cfg['preprocessing'][PREPROCESSING_KARNIG_LIN]['nb_cuts'] = args_parser.nb_cuts
        if args_parser.lb_f is not None:
            main_cfg['preprocessing'][PREPROCESSING_KARNIG_LIN]['lb_f'] = args_parser.lb_f
    elif args_parser.preprocessing_name == PREPROCESSING_FID_MAT:
        if args_parser.nb_cuts is not None:
            main_cfg['preprocessing'][PREPROCESSING_FID_MAT]['nb_cuts'] = args_parser.nb_cuts
        if args_parser.lb_f is not None:
            main_cfg['preprocessing'][PREPROCESSING_FID_MAT]['lb_f'] = args_parser.lb_f

    if args_parser.no_plots is not None:
        main_cfg['plot']['no_plots'] = args_parser.no_plots

    if args_parser.unique_id is not None:
        main_cfg['experiment']['unique_id'] = str(args_parser.unique_id)

    return main_cfg


def load_settings(file):
    with open(file, 'r') as f:
        return yaml.load(f, Loader=yaml.UnsafeLoader)


def validate_names(args):

    try:
        args['experiment']['dataset'] = Dataset(args['experiment']['dataset'])
    except ValueError:
        raise ValueError(f'The dataset name must be in: {Dataset.list()}')

    try:
        args['experiment']['preprocessing'] = Preprocessing(args['experiment']['preprocessing'])
    except ValueError:
        raise ValueError(f'All the preprocessing name must be in: {Preprocessing.list()}')

    try:
        args['experiment']['cut_finding'] = (CutFinding(args['experiment']['cut_finding']))
    except ValueError:
        raise ValueError(f'The cut-finding strategy name must be in: {CutFinding.list()}')

    try:
        args['experiment']['cost_function'] = (CostFunction(args['experiment']['cost_function']))
    except ValueError:
        raise ValueError(f'The cost function name must be in: {CostFunction.list()}')

    return args


def set_up_dirs(args, root_dir):
    args['root_dir'] = Path(root_dir)
    args['output_dir'] = Path(f"{root_dir / 'output' / args['experiment']['unique_id']}")
    args['plot_dir'] = Path(f"{args['output_dir'] / 'plots'}")
    args['answers_dir'] = Path(f"{args['output_dir'] / 'answers'}")

    args['output_dir'].mkdir(parents=True, exist_ok=True)

    return args
