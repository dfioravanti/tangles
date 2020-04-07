from pathlib import Path
from itertools import product

import pandas as pd

from src.parser import make_parser
from src.config import load_validate_settings, set_up_dirs
from src.execution import get_parameters,compute_clusters, compute_evaluation, \
    get_dataset_cuts_order, tangle_computation, plotting
from src.utils import dict_product


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

    all_parameters = get_parameters(args)

    df_output = pd.DataFrame()

    for parameters in dict_product(all_parameters):

        if args.verbose >= 1:
            print(f'Working on parameters = {parameters}', flush=True)

            data, orders, all_cuts = get_dataset_cuts_order(args, parameters)

            tangles_of_order = tangle_computation(args, all_cuts, orders)
            predictions_of_order = compute_clusters(tangles_of_order, all_cuts, verbose=args.verbose)

            evaluation = compute_evaluation(data['ys'], predictions_of_order)
            new_row = pd.Series({**parameters, **evaluation})
            df_output = df_output.append(new_row, ignore_index=True)

            if args.plot.tangles:
                plotting(args, data, predictions_of_order)

    path = args.root_dir / f'evaluation.csv'
    df_output.to_csv(path)


if __name__ == '__main__':

    # Make parser, read inputs from command line and resolve paths

    parser = make_parser()
    args_parser = parser.parse_args()

    args = load_validate_settings(args_parser, root_dir='./')
    root_dir = Path(__file__).resolve().parent
    args = set_up_dirs(args, root_dir=root_dir)

    main(args)
