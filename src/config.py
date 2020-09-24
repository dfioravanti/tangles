from pathlib import Path

import numpy as np
import yaml

from src.my_types import Dataset, Preprocessing, CutFinding, CostFunction

NAN = np.nan


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
