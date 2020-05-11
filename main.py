from pathlib import Path

import json
import pandas as pd

from src.parser import make_parser
from src.config import load_validate_settings, set_up_dirs
from src.execution import compute_clusters, compute_evaluation, get_dataset_cuts_order, tangle_computation, plotting, \
    compute_maximal_tangles, compute_clusters_maximals, print_tangles_names, tangles_to_range_answers, \
    compute_fuzzy_clusters, soft_plotting, centers_in_range_answers  # , compute_soft_evaluation

from src.plotting import plot_heatmap



def main(args):
    """
    Main function of the program.
    The execution is divided in the following steps

       1. Load datasets
       2. Find the bipartions to consider
       3. Compute the order of the bipartions
       4. For each order compute the tangles by expanding on the
          previous ones if it makes sense. It does not make sense to expand
          when we find out that we cannot add all the bipartitions of smaller orders
       5. Plot a heatmap to visually see the clusters

    Parameters
    ----------
    args: SimpleNamespace
       The parameters to the program

    Returns
    -------
    """

    foundamental_parameters = {**args['experiment'], **args['dataset'], **args['preprocessing']}

    unique_id = hash(json.dumps(foundamental_parameters, sort_keys=True))
    df_output = pd.DataFrame()

    if args['verbose'] >= 1:
        print(f'Working with parameters = {foundamental_parameters}', flush=True)

    data, orders, all_cuts, name_cuts = get_dataset_cuts_order(args)
    max_order = orders.max()

    tangles_by_order = tangle_computation(all_cuts=all_cuts,
                                          orders=orders,
                                          agreement=args['experiment']['agreement'],
                                          verbose=args['verbose'])

    # max_tangles = compute_maximal_tangles(tangles_by_order)
    predictions_by_order = compute_clusters(tangles_by_orders=tangles_by_order,
                                            all_cuts=all_cuts,
                                            verbose=args['verbose'])

    soft_predictions_by_order = compute_fuzzy_clusters(tangles_by_orders=tangles_by_order,
                                            all_cuts=all_cuts,
                                            verbose=args['verbose'])

    if data['ys'] is not None:
        evaluation = compute_evaluation(data['ys'], predictions_by_order)
        order_best = evaluation['order_best']
        if args['verbose'] >= 1:
            print(f'Best result \n\t {evaluation}', flush=True)

        new_row = pd.Series({**foundamental_parameters, **evaluation, 'max_order': max_order})
        df_output = df_output.append(new_row, ignore_index=True)
        path = args['root_dir'] / f'evaluation_{unique_id}.csv'
        df_output.to_csv(path)

        #soft_evaluation = compute_soft_evaluation(data['ys'], data['xs'], data['cs'], soft_predictions_by_order)
        #if args['verbose'] >= 1:
        #    print(f'Best result \n\t {soft_evaluation}', flush=True)

    else:
        order_best = None

    if name_cuts is not None:
        range_answers = tangles_to_range_answers(tangles_by_order[order_best], name_cuts,
                                                 interval_values=args['preprocessing']['range_answers'],
                                                 path=args['root_dir'])
        centers_in_range_answers(data['cs'], range_answers)
        print_tangles_names(name_cuts, tangles_by_order, order_best,
                            path=args['answers_dir'],
                            verbose=args['verbose'])

    if args['plot']['tangles']:
        plotting(data, predictions_by_order, verbose=args['verbose'], path=args['plot_dir'])
        soft_plotting(data, soft_predictions_by_order[order_best], path=args['plot_dir'])



if __name__ == '__main__':

    # Make parser, read inputs from command line and resolve paths

    parser = make_parser()
    args_parser = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    print(root_dir)
    args = load_validate_settings(args_parser, root_dir=root_dir)
    args = set_up_dirs(args, root_dir=root_dir)

    main(args)
