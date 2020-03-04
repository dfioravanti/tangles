from functools import partial
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np

from src.config import make_parser, to_SimpleNamespace
from src.loading import get_dataset
from src.execution import compute_cuts, compute_tangles, \
    compute_clusters, order_cuts, compute_evaluation
from src.plotting.questionnaire import plot_tangles_on_questionnaire
from src.acceptance_functions import triplet_size_big_enough


def main(args):

    print("Load data\n", flush=True)
    xs, ys, order_function = get_dataset(args.dataset)

    print("Find cuts", flush=True)
    all_cuts = compute_cuts(xs, args.preprocessing)
    print(f"\tI found {len(all_cuts)} cuts\n", flush=True)

    print("Compute order", flush=True)
    orders = order_cuts(all_cuts, order_function)
    existing_orders = list(orders.keys())
    existing_orders.sort()
    print(f"\tMax order: {existing_orders[-1]} \n", flush=True)

    min_size = np.int(np.floor(0.1 * len(xs)))
    acceptance_function = partial(triplet_size_big_enough, all_cuts=all_cuts, min_size=min_size)
    print(f"Using min_size = {min_size} \n", flush=True)

    predictions = {}
    tangles = []

    print("Start tangle computation", flush=True)
    t_start = datetime.now()

    for idx_order, order in enumerate(existing_orders):

        print(f"\tCompute tangles of order {order}", flush=True)

        cuts_order_i = [orders[j] for j in existing_orders[:idx_order+1]]
        cuts_order_i = [i for sub in cuts_order_i for i in sub]
        tangles = compute_tangles(previous_tangles=tangles, current_cuts=cuts_order_i,
                                  acceptance_function=acceptance_function, algorithm=args.algorithm)

        print(f"\t\tI found {len(tangles)} tangles of order {order}", flush=True)

        print(f"\tCompute clusters for order {order}", flush=True)
        predictions[order] = compute_clusters(tangles, all_cuts)

    t_finish = datetime.now()
    t_total = relativedelta(t_finish, t_start)
    print(f"\nThe computation took {t_total.days} days, {t_total.hours} hours,"
          f" {t_total.minutes} minutes and {t_total.seconds} seconds\n")

    evaluations = compute_evaluation(ys, predictions)

    for order, evaluation in evaluations.items():
        for k, v in evaluation.items():
            print(f"For order {order} we have {k} = {v}", flush=True)

    plot_tangles_on_questionnaire(xs, ys, predictions, path="./plot.pdf")


if __name__ == '__main__':
    from pathlib import Path

    parser = make_parser()
    args = to_SimpleNamespace(parser.parse_args())
    if args.dataset.path is not None:
        args.dataset.path = Path(__file__).resolve().parent / args.dataset.path

    main(args)
