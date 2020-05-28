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

    # Mindsets
    parser.add_argument('--mind_sizes', dest='mind_sizes', nargs='+', type=int)
    parser.add_argument('--mind_questions', dest='mind_questions', action='store', type=int)
    parser.add_argument('--mind_useless', dest='mind_useless', action='store', type=int)
    parser.add_argument('--mind_noise', dest='mind_noise', action='store', type=float)

    # Questionaire
    parser.add_argument('--q_nb_samples', dest='q_nb_samples', action='store', type=int)
    parser.add_argument('--q_nb_features', dest='q_nb_features', action='store', type=int)
    parser.add_argument('--q_nb_mindsets', dest='q_nb_mindsets', action='store', type=int)
    parser.add_argument('--q_range_answers', dest='q_range_answers', nargs='+', type=int)

    # Preprocessing
    parser.add_argument('--nb_cuts', dest='nb_cuts', action='store', type=int)
    parser.add_argument('--lb_f', dest='lb_f', action='store', type=float)

    # ID
    parser.add_argument('--id', dest='unique_id', action='store', default=0)

    # Plotting
    parser.add_argument('--yes_plots', dest='no_plots', action='store_false', default=None)
    parser.add_argument('--no_plots', dest='no_plots', action='store_true', default=None)

    return parser
