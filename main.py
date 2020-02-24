from src.config import make_parser, to_SimpleNamespace
from src.loading import get_dataset
from src.tangles import compute_cuts, compute_tangles, compute_clusters


def main(args):

    xs, ys, cs = get_dataset(args.dataset)
    cuts = compute_cuts(xs, args.preprocessing)
    print(cuts)
    tangles = compute_tangles(cuts)
    ys_predicted = compute_clusters(xs, tangles)


if __name__ == '__main__':

    parser = make_parser()
    args = to_SimpleNamespace(parser.parse_args())

    main(args)
