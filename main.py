from pathlib import Path
from copy import deepcopy
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np

from src.config import load_validate_settings, set_up_dirs
from src.loading import get_dataset_and_order_function
from src.plotting import plot_heatmap_graph, plot_heatmap, plot_cuts
from src.execution import compute_cuts, compute_tangles, order_cuts


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

    print("Load data\n", flush=True)
    xs, ys, G, order_function = get_dataset_and_order_function(args.dataset, args.seed)

    print("Find cuts", flush=True)
    all_cuts = compute_cuts(xs, args.preprocessing)
    print(f"\tI found {len(all_cuts)} cuts\n", flush=True)

    print("Compute order", flush=True)
    all_cuts, orders = order_cuts(all_cuts, order_function)
    max_order = np.int(np.ceil(np.max(orders)))
    min_order = np.int(np.floor(np.min(orders)))
    print(f"\tMax order: {max_order} \n", flush=True)

    if args.plot.cuts:
        if args.dataset.type == 'graph':
            plot_cuts(G, all_cuts[:args.plot.nb_cuts], orders, args.dataset.type, args.output.root_dir)
        else:
            raise NotImplementedError('I still need to implement this')

    min_size = args.algorithm.min_size
    print(f"Using min_size = {min_size} \n", flush=True)

    print("Start tangle computation", flush=True)
    t_start = datetime.now()

    tangles = []
    tangles_of_order = {}
    nb_cuts_considered = 0
    nb_cuts = len(all_cuts)

    for idx_order, order in enumerate(range(min_order, max_order+1)):

        if nb_cuts_considered >= nb_cuts * args.tangles.percentage_cuts:
            break

        idx_cuts_order_i = np.where(np.all([order - 1 < orders, orders <= order],
                                           axis=0))[0]
        nb_cuts_considered += len(idx_cuts_order_i)

        if len(idx_cuts_order_i) > 0:
            print(f"\tCompute tangles of order {order}", flush=True)

            cuts_order_i = all_cuts[idx_cuts_order_i]
            tangles = compute_tangles(tangles, cuts_order_i, idx_cuts_order_i,
                                      min_size=min_size, algorithm=args.algorithm)
            print(f"\t\tI found {len(tangles)} tangles of order {order}", flush=True)
            print(f"\tCompute clusters for order {order}", flush=True)

            tangles_of_order[order] = deepcopy(tangles)

            if tangles == []:
                print(f'Stopped computation at order {order} instead of {max_order}',
                      flush=True)
                break

    t_finish = datetime.now()
    t_total = relativedelta(t_finish, t_start)
    print(f"\nThe computation took {t_total.days} days, {t_total.hours} hours,"
          f" {t_total.minutes} minutes and {t_total.seconds} seconds\n")

    if args.plot.tangles:
        print('Start plotting', flush=True)
        if args.dataset.type == 'graph':
            plot_heatmap_graph(G=G, all_cuts=all_cuts, tangles_by_orders=tangles_of_order, path=args.output.root_dir)
        elif args.dataset.type == 'discrete':
            plot_heatmap(all_cuts=all_cuts, ys=ys, tangles_by_orders=tangles_of_order, path=args.output.root_dir)
        print('Done plotting', flush=True)


if __name__ == '__main__':

    # Make parser, read inputs from command line and resolve paths
    args = load_validate_settings('./')
    root_dir = Path(__file__).resolve().parent
    args = set_up_dirs(args, root_dir=root_dir)

    main(args)
