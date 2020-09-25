import argparse

from src.my_types import Dataset, Preprocessing, CutFinding, CostFunction


def make_parser():
    """
    Build the command line parser for the program.

    Returns
    -------
    parser: ArgumentParser
        The parser

    """

    parser = argparse.ArgumentParser(description='Program to compute tangles')

    parser.add_argument('-v', dest='verbose', action='store', type=int, default=3)

    parser.add_argument('-t', dest='dataset', action='store')
    parser.add_argument('-p', dest='preprocessing', action='store', default='none')
    parser.add_argument('-b', dest='cut_finding', action='store')
    parser.add_argument('-a', dest='agreement', action='store', type=int)
    parser.add_argument('-o', dest='percentile_orders', action='store', type=int)
    parser.add_argument('-s', dest='seed', action='store', type=int)
    parser.add_argument('-c', dest='cost_function', action='store')

    # Cost Function
    parser.add_argument('--sample_cost', dest='sample_cost', action='store', default=-1, type=int)

    # Gaussian
    parser.add_argument('--gauss_sizes', dest='gauss_sizes', nargs='+', type=int)
    parser.add_argument('--gauss_mean', dest='gauss_centers', nargs='+', type=int)
    parser.add_argument('--gauss_var', dest='gauss_variances', nargs='+', type=float)

    # Mindsets
    parser.add_argument('--mind_sizes', dest='mind_sizes', nargs='+', type=int)
    parser.add_argument('--mind_questions', dest='mind_questions', action='store', type=int)
    parser.add_argument('--mind_useless', dest='mind_useless', action='store', type=int)
    parser.add_argument('--mind_noise', dest='mind_noise', action='store', type=float)

    # Moons
    parser.add_argument('--moon_n', dest='n_smaples', type=int)
    parser.add_argument('--moon_rad', dest='moon_radius', action='store', type=float)
    parser.add_argument('--moon_noise', dest='moon_noise', action='store', type=float)

    # SBM
    parser.add_argument('--sbm_sizes', dest='block_sizes', nargs='+', type=int)
    parser.add_argument('--sbm_p', dest='sbm_p', action='store', type=float)
    parser.add_argument('--sbm_q', dest='sbm_q', action='store', type=float)

    # Questionaire
    parser.add_argument('--q_nb_samples', dest='q_nb_samples', action='store', type=int)
    parser.add_argument('--q_nb_features', dest='q_nb_features', action='store', type=int)
    parser.add_argument('--q_nb_mindsets', dest='q_nb_mindsets', action='store', type=int)
    parser.add_argument('--q_centers', dest='q_centers', type=int, default=False)
    parser.add_argument('--q_range_answers', dest='q_range_answers', nargs='+', type=int)

    # big5
    parser.add_argument('--path', dest='big5_path', nargs='+', type=int)

    # Preprocessing
    parser.add_argument('--k', dest='k', action='store', type=int)
    parser.add_argument('--rad', dest='radius', action='store', type=float)

    # Cut finding
    parser.add_argument('--nb_cuts', dest='nb_cuts', action='store', type=int)

    # random projection
    parser.add_argument('--proj_dime', dest='dimension_random_project', action='store', type=int, default=1)

    # bin
    parser.add_argument('--n_bins', dest='n_bins', action='store', type=int)

    # kl and fm
    parser.add_argument('--lb_f', dest='lb_f', action='store', type=float)
    parser.add_argument('--early_stopping', dest='early_stopping', action='store_true', default=False)

    # ID
    parser.add_argument('--id', dest='unique_id', action='store', default=0)

    # Plotting
    parser.add_argument('--plots', dest='no_plots', action='store_false', default=True)

    parser.add_argument('--plot_tree', dest='plot_tree', action='store_true', default=False)
    parser.add_argument('--plot_tangles', dest='plot_tangles', action='store_true', default=False)
    parser.add_argument('--plot_cuts', dest='plot_cuts', action='store_true', default=False)
    parser.add_argument('--plot_nb_cuts', dest='plot_nb_cuts', action='store', default=10, type=int)
    parser.add_argument('--plot_soft', dest='plot_soft', action='store_true', default=False)
    parser.add_argument('--plot_hard', dest='plot_hard', action='store_true', default=False)

    return parser


def get_arguments(cmd_args):
    cmd_args = vars(cmd_args)
    if cmd_args['dataset'] == None:
        return None

    args = {'experiment': {}, 'dataset': {}, 'preprocessing': {}, 'cut_finding': {}, 'plot': {}, 'verbose': cmd_args['verbose']}

    try:
        args['experiment']['dataset'] = Dataset(cmd_args['dataset'])
    except ValueError:
        raise ValueError(f'The dataset name must be in: {Dataset.list()}')

    try:
        args['experiment']['preprocessing'] = Preprocessing(cmd_args['preprocessing'])
    except ValueError:
        raise ValueError(f'All the preprocessing name must be in: {Preprocessing.list()}')

    try:
        args['experiment']['cut_finding'] = (CutFinding(cmd_args['cut_finding']))
    except ValueError:
        raise ValueError(f'The cut-finding strategy name must be in: {CutFinding.list()}')

    try:
        args['experiment']['cost_function'] = (CostFunction(cmd_args['cost_function']))
    except ValueError:
        raise ValueError(f'The cost function name must be in: {CostFunction.list()}')

    args['experiment']['unique_id'] = cmd_args['unique_id']
    args['experiment']['seed'] = cmd_args['seed']
    args['experiment']['agreement'] = cmd_args['agreement']
    args['experiment']['percentile_orders'] = cmd_args['percentile_orders']

    # preprocessing
    if cmd_args['preprocessing'] == 'knn' or cmd_args['preprocessing'] == 'wknn':
        args['preprocessing']['k'] = cmd_args['k']
    elif cmd_args['preprocessing'] == 'rng':
        args['preprocessing']['radius'] = cmd_args['radius']

    # cost function
    args['cost_function'] = {'nb_points': cmd_args['sample_cost']}

    # dataset
    if cmd_args['dataset'] == 'gau_mix':
        args['dataset']['sizes'] = cmd_args['gauss_sizes']
        args['dataset']['centers'] = cmd_args['gauss_centers']
        args['dataset']['variances'] = cmd_args['gauss_variances']

    elif cmd_args['dataset'] == 'mind':
        args['dataset']['mindset_sizes'] = cmd_args['mind_sizes']
        args['dataset']['nb_questions'] = cmd_args['mind_questions']
        args['dataset']['nb_useless'] = cmd_args['mind_useless']
        args['dataset']['noise'] = cmd_args['mind_noise']

    elif cmd_args['dataset'] == 'moons':
        args['dataset']['n_samples'] = cmd_args['n_smaples']
        args['dataset']['noise'] = cmd_args['moon_noise']
        args['dataset']['radius'] = cmd_args['moon_radius']

    elif cmd_args['dataset'] == 'sbm':
        args['dataset']['block_sizes'] = cmd_args['block_sizes']
        args['dataset']['p'] = cmd_args['sbm_p']
        args['dataset']['q'] = cmd_args['sbm_q']

    elif cmd_args['dataset'] == 'qst_lkr':
        args['dataset']['nb_samples'] = cmd_args['q_nb_samples']
        args['dataset']['nb_features'] = cmd_args['q_nb_features']
        args['dataset']['nb_mindsets'] = cmd_args['q_nb_mindsets']
        args['dataset']['centers'] = cmd_args['q_centers']
        args['dataset']['range_answers'] = cmd_args['q_range_answers']

    elif cmd_args['dataset'] == 'BCW':
        pass

    elif cmd_args['dataset'] == 'big5':
        args['dataset']['path'] = cmd_args['big5_path']

    else:
        ValueError("One of the above must apply!")

    # preprocessing
    if cmd_args['cut_finding'] == 'rand_proj':
        args['cut_finding']['nb_cuts'] = cmd_args['nb_cuts']
        args['cut_finding']['dimension'] = cmd_args['dimension_random_project']
    elif cmd_args['cut_finding'] == 'KL' or cmd_args['cut_finding'] == 'FM':
        args['cut_finding']['nb_cuts'] = cmd_args['nb_cuts']
        args['cut_finding']['lb_f'] = cmd_args['lb_f']
        args['cut_finding']['early_stopping'] = cmd_args['early_stopping']
    elif cmd_args['cut_finding'] == 'bins':
        args['cut_finding']['n_bins'] = cmd_args['n_bins']

    # plotting
    args['plot']['no_plots'] = cmd_args['no_plots']
    args['plot']['tree'] = cmd_args['plot_tree']
    args['plot']['tangles'] = cmd_args['plot_tangles']
    args['plot']['cuts'] = cmd_args['plot_cuts']
    args['plot']['nb_cuts'] = cmd_args['plot_nb_cuts']
    args['plot']['soft'] = cmd_args['plot_soft']
    args['plot']['hard'] = cmd_args['plot_hard']

    return args
