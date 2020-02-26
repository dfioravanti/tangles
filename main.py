from matplotlib import pyplot as plt

from src.config import make_parser, to_SimpleNamespace
from src.loading import get_dataset
from src.execution import compute_cuts, compute_tangles, compute_clusters
from src.plotting.questionnaire import plot_tangles_on_questionnaire


def main(args):

    xs, ys, cs = get_dataset(args.dataset)
    cuts = compute_cuts(xs, args.preprocessing)
    tangles = compute_tangles(xs, cuts, args.algorithm)
    masks_tangles = compute_clusters(xs, tangles)

    # TODO: Figure out why I get different positions for the same x!
    plot_tangles_on_questionnaire(xs, ys, masks_tangles)


if __name__ == '__main__':

    parser = make_parser()
    args = to_SimpleNamespace(parser.parse_args())

    main(args)
