from matplotlib import pyplot as plt
import numpy as np

from src.config import make_parser, to_SimpleNamespace
from src.loading import get_dataset
from src.execution import compute_cuts, compute_tangles, compute_clusters, order_cuts
from src.plotting.questionnaire import plot_tangles_on_questionnaire


def main(args):

    xs, ys, order_function = get_dataset(args.dataset)
    cuts = compute_cuts(xs, args.preprocessing)
    orders = order_cuts(cuts, order_function)
    tangles = compute_tangles(xs, cuts[orders[2]], args.algorithm)
    masks_tangles = compute_clusters(xs, tangles)

    plot_tangles_on_questionnaire(xs, ys, masks_tangles)


if __name__ == '__main__':

    parser = make_parser()
    args = to_SimpleNamespace(parser.parse_args())

    main(args)
