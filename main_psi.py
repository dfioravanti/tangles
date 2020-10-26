from copy import deepcopy
from pathlib import Path

from sklearn.metrics import normalized_mutual_info_score
from sklearn.neighbors._dist_metrics import DistanceMetric

from src.config import load_validate_parser, set_up_dirs, load_validate_config_file, deactivate_plots
from src.execution import compute_and_save_evaluation, tangle_computation, \
    compute_soft_predictions, compute_hard_predictions, save_time_evaluation, get_data, get_cuts, \
    compute_mindset_prediciton, find_bipartitions, get_cost_function, compute_cost_and_order_cuts, pick_cuts_up_to_order
from src.baselines import compute_and_save_comparison
from src.my_types import Dataset
from src.parser import make_parser
from src.plotting import plot_soft_predictions, plot_hard_predictions
from datetime import datetime
import time
import numpy as np

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

    data = get_data(args)

    seed = args['experiment']['seed']

    for r in range(1, args['runs']+1):
        args['experiment']['seed'] = r + seed
        hyperparameters['seed'] = args['experiment']['seed']

        if args['verbose'] >= 2:
            print("Find cuts", flush=True)

        bipartitions, _ = find_bipartitions(data, args, verbose=args['verbose'])

        if args['verbose'] >= 2:
            print('\tI found {} cuts\n'.format(len(bipartitions.values)))

        print("Compute cost", flush=True)
        cost_function = get_cost_function(data, args)

        bipartitions = compute_cost_and_order_cuts(bipartitions, cost_function)

        cuts = pick_cuts_up_to_order(deepcopy(bipartitions),
                                                 percentile=args['experiment']['percentile_orders'])
        print("number of cuts: ", len(cuts.values))
        if args['verbose'] >= 2:
            max_considered_order = cuts.costs[-1]
            print("\tI will stop at order: {}".format(max_considered_order))
            print('\tI will use {} cuts\n'.format(len(cuts.values)), flush=True)


        tangles_tree = tangle_computation(bipartitions=cuts,
                                                  agreement=args['experiment']['agreement'],
                                                  verbose=args['verbose'])

        contracted_tree = ContractedTangleTree(tangles_tree, prune_depth=args['experiment']['prune_depth'])

        if args['experiment']['dataset'] == Dataset.mindsets:
            ys_predicted, cs = compute_hard_predictions(contracted_tree,
                                                       cuts=bipartitions, xs=data.xs)
        else:
            ys_predicted, _ = compute_hard_predictions(contracted_tree,
                                                           cuts=bipartitions)
            print("All good!. Used this for postprocessing.")

        if data.ys is not None:
            compute_and_save_evaluation(ys=data.ys,
                                                ys_predicted=ys_predicted,
                                                hyperparameters=hyperparameters,
                                                id_run=id_run,
                                                path=args['output_dir'],
                                                r=r)

# def pick_cuts_up_to_order(bipartitions, percentile):
#     """
#     Drop the cuts whose order is in a percentile above percentile.
#
#     Parameters
#     ----------
#     cuts: Bipartitions
#     percentile
#
#     Returns
#     -------
#     """
#
#     mask_orders_to_pick = bipartitions.costs <= np.percentile(bipartitions.costs[~np.isnan(bipartitions.costs)], q=percentile)
#     bipartitions.costs = bipartitions.costs[mask_orders_to_pick]
#     bipartitions.values = bipartitions.values[mask_orders_to_pick, :]
#     if bipartitions.names is not None:
#         bipartitions.names = bipartitions.names[mask_orders_to_pick]
#     if bipartitions.equations is not None:
#         bipartitions.equations = bipartitions.equations[mask_orders_to_pick]
#
#     return bipartitions

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
