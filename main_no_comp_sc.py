from pathlib import Path

from src.config import load_validate_parser, set_up_dirs, load_validate_config_file, deactivate_plots
from src.execution import compute_and_save_evaluation, get_data_and_cuts, tangle_computation, \
    compute_soft_predictions, compute_hard_preditions, save_time_evaluation, compute_and_save_comparison
from src.parser import make_parser
from src.plotting import plot_soft_predictions, plot_hard_predictions
from datetime import datetime
import time

from src.tree_tangles import ContractedTangleTree


def main(args):
    """
    Main function of the program.
    The execution is divided in the following steps

       1. Load datasets
       2. Find the cuts and compute the costs
       3. For each cut compute the tangles by expanding on the
          previous ones if it is consistent. If its not possible stop
       4. Postprocess in soft and hard clustering 

    Parameters
    ----------
    args: SimpleNamespace
       The parameters to the program

    Returns
    -------
    """

    hyperparameters = {**args['experiment'], **args['dataset'], **args['preprocessing'], **args['cut_finding']}
    id_run = datetime.now().strftime("%m-%d")

    if args['verbose'] >= -1:
        print('ID for the run = {}'.format(id_run))
        print('Working with hyperparameters = {}'.format(hyperparameters))
        print('Plot settings = {}'.format(args["plot"]), flush=True)

    if args['runs'] > 1:
        args['verbose'] = 0
        args['plot']['no_plots'] = True
        deactivate_plots(args)

    for r in range(1, args['runs']+1):
        start_all = time.time()
        args['experiment']['seed'] = r * args['experiment']['seed']
        data, bipartitions, preprocessing_time, cost_and_sort_time = get_data_and_cuts(args)

        start = time.time()
        tangles_tree = tangle_computation(bipartitions=bipartitions,
                                          agreement=args['experiment']['agreement'],
                                          verbose=args['verbose'])

        tangle_search_tree_time = time.time() - start

        start = time.time()
        contracted_tree = ContractedTangleTree(tangles_tree)

        compute_soft_predictions(contracted_tree=contracted_tree,
                                 cuts=bipartitions,
                                 verbose=args['verbose'])
        soft_clustering_time = time.time() - start

        time_all = time.time() - start_all

        save_time_evaluation(id_run=id_run,
                             pre_time=preprocessing_time,
                             cost_time=cost_and_sort_time,
                             tst_time=tangle_search_tree_time,
                             post_time=soft_clustering_time,
                             all_time=time_all,
                             path=args['output_dir'],
                             verbose=args['verbose'],
                             r=r)

        if args['plot']['tree']:
            tangles_tree.plot_tree(path=args['output_dir'] / 'tree.svg')

        if args['plot']['tree']:
            contracted_tree.plot_tree(path=args['output_dir'] / 'contracted.svg')

        if args['plot']['soft']:
            path = args['output_dir'] / 'clustering'
            plot_soft_predictions(data=data,
                                  contracted_tree=contracted_tree,
                                  eq_cuts=bipartitions.equations,
                                  path=path)

        ys_predicted = compute_hard_preditions(contracted_tree,
                                               cuts=bipartitions)

        if args['plot']['hard']:
            path = args['output_dir'] / 'clustering'
            plot_hard_predictions(data, ys_predicted, path=path)

        if data.ys is not None:
            compute_and_save_evaluation(ys=data.ys,
                                        ys_predicted=ys_predicted,
                                        hyperparameters=hyperparameters,
                                        id_run=id_run,
                                        path=args['output_dir'],
                                        r=r)

if __name__ == '__main__':

    # Make parser and parse command line
    parser = make_parser()
    args_parser = parser.parse_args()
    args = load_validate_parser(args_parser)

    root_dir = Path(__file__).resolve().parent
    if args is None:
        cfg_file_path = root_dir / 'settings.yml'
        args = load_validate_config_file(cfg_file_path)

    args = deactivate_plots(args)
    args = set_up_dirs(args, root_dir=root_dir)

    main(args)
