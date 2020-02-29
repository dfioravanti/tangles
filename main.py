from matplotlib import pyplot as plt
import numpy as np

from src.config import make_parser, to_SimpleNamespace
from src.loading import get_dataset
from src.execution import compute_cuts, compute_tangles, compute_clusters, order_cuts, fix_indexes
from src.plotting.questionnaire import plot_tangles_on_questionnaire


def main(args):

    print("Load data")
    xs, ys, order_function = get_dataset(args.dataset)

    print("Find cuts")
    cuts = compute_cuts(xs, args.preprocessing)
    print(f"I found {len(cuts)} cuts")

    print("Compute order")
    orders = order_cuts(cuts, order_function)

    existing_orders = list(orders.keys())
    existing_orders.sort()
    print(f"Max order: {existing_orders[-1]}")

    masks = []

    for i in range(len(existing_orders)):

        print(f"Compute tangles of order {existing_orders[i]}")

        current_cuts = [orders[j] for j in existing_orders[:i+1]]
        current_cuts = [i for sub in current_cuts for i in sub]
        tangles = compute_tangles(xs, cuts[current_cuts], args.algorithm)
        fix_indexes(tangles, current_cuts)

        print("Compute clusters")
        masks.append(compute_clusters(xs, tangles, cuts))

    plot_tangles_on_questionnaire(xs, ys, masks, existing_orders)


if __name__ == '__main__':

    parser = make_parser()
    args = to_SimpleNamespace(parser.parse_args())

    main(args)
