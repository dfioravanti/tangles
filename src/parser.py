import argparse


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
    parser.add_argument('-p', dest='preprocessing_name', action='store')
    parser.add_argument('-a', dest='agreement', action='store', type=int)
    parser.add_argument('-o', dest='percentile_orders', action='store', type=int)
    parser.add_argument('-s', dest='seed', action='store', type=int)

    # SBM
    parser.add_argument('--sbm_bs', dest='sbm_bs', nargs='+', type=int)
    parser.add_argument('--sbm_p', dest='sbm_p', action='store', type=float)
    parser.add_argument('--sbm_q', dest='sbm_q', action='store', type=float)

    # Gaussian + KNN
    parser.add_argument('--gauss_bs', dest='gauss_bs', nargs='+', type=int)
    parser.add_argument('--gauss_cs', dest='gauss_cs', nargs='+', type=float)
    parser.add_argument('--gauss_k', dest='gauss_k', action='store', type=int)

    parser.add_argument('--nb_cuts', dest='nb_cuts', action='store', type=int)
    parser.add_argument('--lb_f', dest='lb_f', action='store', type=int)

    # Plotting
    parser.add_argument('--plot_tangles', dest='plot_tangles', action='store_true', default=None)
    parser.add_argument('--no_plot_tangles', dest='plot_tangles', action='store_false', default=None)

    parser.add_argument('--plot_cuts', dest='plot_cuts', action='store_true', default=None)
    parser.add_argument('--no_plot_cuts', dest='plot_cuts', action='store_false', default=None)

    return parser
