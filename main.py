from pathlib import Path

from sklearn.metrics import normalized_mutual_info_score
from sklearn.neighbors._dist_metrics import DistanceMetric

from src.config import load_validate_parser, set_up_dirs, load_validate_config_file, deactivate_plots
from src.execution import compute_and_save_evaluation, tangle_computation, \
    compute_soft_predictions, compute_hard_predictions, save_time_evaluation, get_data, get_cuts, \
    compute_mindset_prediciton
from src.baselines import compute_and_save_comparison
from src.my_types import Dataset
from src.parser import make_parser
from src.plotting import plot_soft_predictions, plot_hard_predictions
from datetime import datetime
import time
import numpy as np

from src.tree_tangles import ContractedTangleTree
import matplotlib.pyplot as plt


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

    data = get_data(args)

    seed = args['experiment']['seed']

    for r in range(1, args['runs']+1):
        start_all = time.time()
        args['experiment']['seed'] = r + seed
        hyperparameters['seed'] = args['experiment']['seed']
        bipartitions, preprocessing_time, cost_and_sort_time = get_cuts(args, data)

        start = time.time()
        tangles_tree = tangle_computation(bipartitions=bipartitions,
                                          agreement=args['experiment']['agreement'],
                                          verbose=args['verbose'])

        tangle_search_tree_time = time.time() - start

        start = time.time()
        contracted_tree = ContractedTangleTree(tangles_tree, prune_depth=args['experiment']['prune_depth'])

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

        if args['experiment']['dataset'] == Dataset.mindsets:
            ys_predicted, cs = compute_hard_predictions(contracted_tree,
                                               cuts=bipartitions, xs=data.xs)

            metric = DistanceMetric.get_metric('manhattan')

            distance = metric.pairwise(cs, data.cs)

            print([np.min(d) for d in distance])

            ys_predicted_gt = compute_mindset_prediciton(data.xs, data.cs)


            print("ground truth: ", normalized_mutual_info_score(data.ys, ys_predicted_gt))
        else:
            ys_predicted, _ = compute_hard_predictions(contracted_tree,
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

    compute_and_save_comparison(data=data,
                                hyperparameters=hyperparameters,
                                id_run=id_run,
                                path=args['output_dir'],
                                r=1)


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
