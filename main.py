from functools import partial

from src.config import make_parser, to_SimpleNamespace
from src.loading import get_dataset
from src.execution import compute_cuts, compute_tangles, compute_clusters, order_cuts
from src.plotting.questionnaire import plot_tangles_on_questionnaire
from src.acceptance_functions import size_big_enough


def main(args):

    print("Load data")
    xs, ys, order_function = get_dataset(args.dataset)

    print("Find cuts")
    cuts = compute_cuts(xs, args.preprocessing)
    print(f"I found {len(cuts)} cuts")
    acceptance_function = partial(size_big_enough, cuts, 30)

    print("Compute order")
    orders = order_cuts(cuts, order_function)

    existing_orders = list(orders.keys())
    existing_orders.sort()
    print(f"Max order: {existing_orders[-1]}")

    masks = []
    tangles = []

    for i in range(len(existing_orders)):

        print(f"Compute tangles of order {i}")

        cuts_order_i = [orders[j] for j in existing_orders[:i+1]]
        cuts_order_i = [i for sub in cuts_order_i for i in sub]
        tangles = compute_tangles(algorithm=args.algorithm, xs=xs,
                                  cuts=cuts_order_i, previous_tangles=tangles,
                                  acceptance_function=acceptance_function)

        print("Compute clusters")
        masks.append(compute_clusters(tangles, cuts))

    plot_tangles_on_questionnaire(xs, ys, masks, existing_orders)


if __name__ == '__main__':

    parser = make_parser()
    args = to_SimpleNamespace(parser.parse_args())

    main(args)
