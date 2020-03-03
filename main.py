from functools import partial

import numpy as np

from src.config import make_parser, to_SimpleNamespace
from src.loading import get_dataset
from src.execution import compute_cuts, compute_tangles, \
    compute_clusters, order_cuts, compute_evaluation
from src.plotting.questionnaire import plot_tangles_on_questionnaire
from src.acceptance_functions import triplet_size_big_enough


def main(args):

    print("Load data", flush=True)
    xs, ys, order_function = get_dataset(args.dataset)

    print("Find cuts", flush=True)
    all_cuts = compute_cuts(xs, args.preprocessing)
    print(f"I found {len(all_cuts)} cuts", flush=True)
    min_size = np.int(np.floor(0.2 * len(xs)))
    acceptance_function = partial(triplet_size_big_enough, all_cuts=all_cuts, min_size=min_size)

    print("Compute order", flush=True)
    orders = order_cuts(all_cuts, order_function)

    existing_orders = list(orders.keys())
    existing_orders.sort()
    print(f"Max order: {existing_orders[-1]}", flush=True)

    predictions = {}
    tangles = []

    for idx_order, order in enumerate(existing_orders):

        print(f"Compute tangles of order {order}", flush=True)

        cuts_order_i = [orders[j] for j in existing_orders[:idx_order+1]]
        cuts_order_i = [i for sub in cuts_order_i for i in sub]
        tangles = compute_tangles(previous_tangles=tangles, current_cuts=cuts_order_i,
                                  acceptance_function=acceptance_function, algorithm=args.algorithm)

        print(f"I found {len(tangles)} tangles of order {order}", flush=True)

        print("Compute clusters")
        predictions[order] = compute_clusters(tangles, all_cuts)

    evaluations = compute_evaluation(ys, predictions)

    for order, evaluation in evaluations.items():
        for k, v in evaluation.items():
            print(f"For order {order} we have {k} = {v}", flush=True)

    plot_tangles_on_questionnaire(xs, ys, predictions)


if __name__ == '__main__':

    parser = make_parser()
    args = to_SimpleNamespace(parser.parse_args())

    main(args)
