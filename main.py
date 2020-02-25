from matplotlib import pyplot as plt

from src.config import make_parser, to_SimpleNamespace
from src.loading import get_dataset
from src.execution import compute_cuts, compute_tangles, compute_clusters
from src.plotting.questionnaire import plot_questionnaire


def main(args):

    xs, ys, cs = get_dataset(args.dataset)
    cuts = compute_cuts(xs, args.preprocessing)
    tangles = compute_tangles(xs, cuts, args.algorithm)
    ys_predicted = compute_clusters(xs, tangles)
    fig, axs = plt.subplots(1, 2, figsize=(7, 7))

    # TODO: Figure out why I get different positions for the same x!
    plot_questionnaire(xs, ys, ax=axs[0])
    plot_questionnaire(xs, ys_predicted[0], ax=axs[1])
    plt.show()


if __name__ == '__main__':

    parser = make_parser()
    args = to_SimpleNamespace(parser.parse_args())

    main(args)
