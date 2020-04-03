from pathlib import Path

import pandas as pd

from src.config import EXPERIMENT_BATCH, EXPERIMENT_SINGLE
from src.config import load_validate_settings, set_up_dirs
from src.execution import compute_clusters, compute_evaluation, get_dataset_cuts_order, tangle_computation, plotting


def main_single_experiment(args):
    """
    Main function of the program.
    The execution is divided in the following steps

        1. Load datasets
        2. Find the bipartions to consider
        3. Compute the order of the bipartions
        4. For each order compute the tangles by expanding on the
           previous ones if it makes sense. It does not make sense to expand
           when we find out that we cannot add all the bipartitions of smaller orders
        5. Plot a heatmap to visually see the clusters

    Parameters
    ----------
    args: SimpleNamespace
        The parameters to the program

    Returns
    -------

    """

    xs, ys, G, orders, all_cuts = get_dataset_cuts_order(args)

    tangles_of_order = tangle_computation(args, all_cuts, orders)
    predictions = compute_clusters(tangles_of_order, all_cuts)

    evaluation = compute_evaluation(ys, predictions)
    print(evaluation)

    if args.plot.tangles:
        plotting(args, predictions, G, ys, all_cuts)


def main_batch_experiment(args):
    for nb_blocks in args.experiment.sbm.nbs_blocks:

        print(f'Working on nb_blocks = {nb_blocks}', flush=True)

        homogeneity = pd.DataFrame()
        completeness = pd.DataFrame()
        v_measure_score = pd.DataFrame()
        unassigned = pd.DataFrame()

        for p in args.experiment.sbm.ps:
            nb_repetitions = args.nb_repetitions
            for i in range(nb_repetitions):
                h, c, v, u = [], [], [], []
                qs = args.experiment.sbm.qs

                for q in qs:
                    print(f'\tWorking with ({p}, {q}): {i + 1}/{nb_repetitions}', flush=True)

                    args.dataset.sbm.block_size = args.experiment.sbm.block_sizes
                    args.dataset.sbm.nb_blocks = nb_blocks
                    args.dataset.sbm.p = p
                    args.dataset.sbm.q = q

                    args.preprocessing.karnig_lin.fractions = args.preprocessing.karnig_lin.fractions[:(nb_blocks + 1)]

                    xs, ys, G, orders, all_cuts = get_dataset_cuts_order(args)
                    tangles_of_order = tangle_computation(args, all_cuts, orders)
                    predictions = compute_clusters(tangles_of_order, all_cuts)

                    evaluation = compute_evaluation(ys, predictions)

                    h.append(evaluation["homogeneity"])
                    c.append(evaluation["completeness"])
                    v.append(evaluation["v_measure_score"])
                    u.append(evaluation['unassigned'])

                h = pd.DataFrame([h], columns=qs, index=[p])
                c = pd.DataFrame([c], columns=qs, index=[p])
                v = pd.DataFrame([v], columns=qs, index=[p])
                u = pd.DataFrame([u], columns=qs, index=[p])

                homogeneity = homogeneity.append(h)
                completeness = completeness.append(c)
                v_measure_score = v_measure_score.append(v)
                unassigned = unassigned.append(u)

        path = args.output.dir / f'nb_blocks_{nb_blocks}_homogeneity.csv'
        homogeneity.to_csv(path)

        path = args.output.dir / f'nb_blocks_{nb_blocks}_completeness.csv'
        completeness.to_csv(path)

        path = args.output.dir / f'nb_blocks_{nb_blocks}_v_measure_score.csv'
        v_measure_score.to_csv(path)

        path = args.output.dir / f'nb_blocks_{nb_blocks}_unassigned.csv'
        unassigned.to_csv(path)


if __name__ == '__main__':

    # Make parser, read inputs from command line and resolve paths
    args = load_validate_settings('./')
    root_dir = Path(__file__).resolve().parent
    args = set_up_dirs(args, root_dir=root_dir)

    if args.experiment.type == EXPERIMENT_SINGLE:
        main_single_experiment(args)
    elif args.experiment.type == EXPERIMENT_BATCH:
        main_batch_experiment(args)
