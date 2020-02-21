import argparse

from src.loading import get_dataset
from src.tangles import compute_cuts, compute_tangles, compute_clusters


def make_parser():
    parser = argparse.ArgumentParser(description='Program to compute tangles')

    return parser


def main():

    xs, ys, cs = get_dataset()
    cuts = compute_cuts(xs)
    tangles = compute_tangles(cuts)
    ys_predicted = compute_clusters(xs, tangles)


if __name__ == '__main__':

    parser = make_parser()
    args = parser.parse_args()

    main()