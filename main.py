from pathlib import Path
from copy import deepcopy
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np

from src.config import make_parser, to_SimpleNamespace
from src.loading import get_dataset
from src.plotting import plot_heatmap, plot_dataset
from src.execution import compute_cuts, compute_tangles, order_cuts


def main(args):

    path_plot = Path("./plots")
    path_plot.mkdir(exist_ok=True)

    print("Load data\n", flush=True)
    xs, ys, order_function = get_dataset(args.dataset)
    plot_dataset(xs, ys, path=path_plot)

    print("Find cuts", flush=True)
    all_cuts = compute_cuts(xs, args.preprocessing)
    print(f"\tI found {len(all_cuts)} cuts\n", flush=True)

    print("Compute order", flush=True)
    all_cuts, orders = order_cuts(all_cuts, order_function)
    max_order = np.int(np.ceil(np.max(orders)))

    print(f"\tMax order: {max_order} \n", flush=True)

    min_size = int(len(xs) * .20)
    print(f"Using min_size = {min_size} \n", flush=True)

    print("Start tangle computation", flush=True)
    t_start = datetime.now()

    tangles = []
    tangles_of_order = {}

    for idx_order, order in enumerate(range(max_order)):

        idx_cuts_order_i = np.where(np.all([order - 1 < orders, orders <= order], axis=0))[0]
        if len(idx_cuts_order_i) > 0:
            print(f"\tCompute tangles of order {order}", flush=True)

            cuts_order_i = all_cuts[idx_cuts_order_i]
            tangles = compute_tangles(tangles, cuts_order_i, idx_cuts_order_i,
                                      min_size=min_size, algorithm=args.algorithm)
            print(f"\t\tI found {len(tangles)} tangles of order {order}", flush=True)
            print(f"\tCompute clusters for order {order}", flush=True)
            tangles_of_order[order] = deepcopy(tangles)

    t_finish = datetime.now()
    t_total = relativedelta(t_finish, t_start)
    print(f"\nThe computation took {t_total.days} days, {t_total.hours} hours,"
          f" {t_total.minutes} minutes and {t_total.seconds} seconds\n")

    plot_heatmap(all_cuts, ys, tangles_of_order, path=path_plot)


if __name__ == '__main__':

    parser = make_parser()
    args = to_SimpleNamespace(parser.parse_args())
    if args.dataset.path is not None:
        args.dataset.path = Path(__file__).resolve().parent / args.dataset.path

    main(args)
