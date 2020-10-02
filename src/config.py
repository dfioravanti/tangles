from pathlib import Path

import numpy as np
import yaml

from src.execution import clean_str
from src.my_types import Dataset, Preprocessing, CutFinding, CostFunction
from src.parser import get_arguments

NAN = np.nan


def load_validate_parser(cmd_args):
    args = get_arguments(cmd_args)

    if args is not None:
        args = validate_settings(args, mode='command_line')

    return args


def delete_useless_parameters(args):

    for operation in ['dataset', 'preprocessing', 'cut_finding']:

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

    print(args['experiment'])

    try:
        args['experiment']['dataset'] = Dataset(args['experiment']['dataset'])
    except ValueError:
        raise ValueError('The dataset name must be in: {}'.format(Dataset.list()))

    try:
        args['experiment']['preprocessing'] = Preprocessing(args['experiment']['preprocessing'])
    except ValueError:
        raise ValueError('All the preprocessing name must be in: {}'.format(Preprocessing.list()))

    try:
        args['experiment']['cut_finding'] = (CutFinding(args['experiment']['cut_finding']))
    except ValueError:
        raise ValueError('The cut-finding strategy name must be in: {}'.format(CutFinding.list()))

    try:
        args['experiment']['cost_function'] = (CostFunction(args['experiment']['cost_function']))
    except ValueError:
        raise ValueError('The cost function name must be in: {}'.format(CostFunction.list()))

    return args


def set_up_dirs(args, root_dir):
    hyperparams = {**args['experiment'], **args['preprocessing'], **args['cut_finding']}

    del hyperparams['unique_id']
    del hyperparams['dataset']

    args['root_dir'] = Path(root_dir)
    args['output_dir'] = Path("{}".format(root_dir / 'output' / str(args['experiment']['dataset']) / str(args['experiment']['unique_id']) / "-".join(map(clean_str, [hyperparams[key] for key in sorted(hyperparams.keys())]))))
    args['plot_dir'] = Path("{}".format(args['output_dir'] / 'plots'))
    args['answers_dir'] = Path("{}".format(args['output_dir'] / 'answers'))

    args['output_dir'].mkdir(parents=True, exist_ok=True)

    return args
