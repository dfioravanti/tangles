from pathlib import Path

from src.config import load_validate_parser, set_up_dirs, load_validate_config_file, deactivate_plots
from src.execution import compute_and_save_evaluation, get_data_and_cuts, tangle_computation, \
    compute_soft_predictions, compute_hard_preditions
from src.parser import make_parser
from src.plotting import plot_soft_predictions, plot_hard_predictions
from src.tree_tangles import contract_tree
from src.utils import get_hash


def main(args):
    """
    Main function of the program.
    The execution is divided in the following steps

       1. Load datasets
       2. Find the cuts and compute the costs
       3. For each cut compute the tangles by expanding on the
          previous ones if it is consistent. If its not possible stop
       4. Postprocess in soft and hard clustering 

    Parameters
    ----------
    args: SimpleNamespace
       The parameters to the program

    Returns
    -------
    """

    hyperparameters = {**args['experiment'], **args['dataset'], **args['preprocessing']}
    id_run = get_hash(hyperparameters)

    if args['verbose'] >= 1:
        print(f'ID for the run = {id_run}')
        print(f'Working with hyperparameters = {hyperparameters}')
        print(f'Plot settings = {args["plot"]}', flush=True)

    data, cuts = get_data_and_cuts(args)

    tangles_tree = tangle_computation(cuts=cuts,
                                      agreement=args['experiment']['agreement'],
                                      verbose=args['verbose'])

    if args['plot']['tree']:
        tangles_tree.plot_tree(path=args['output_dir'] / 'tree.svg')

    contracted_tree = contract_tree(tangles_tree)
    if args['plot']['tree']:
        contracted_tree.plot_tree(path=args['output_dir'] / 'contracted.svg')

    compute_soft_predictions(contracted_tree=contracted_tree,
                             cuts=cuts,
                             verbose=args['verbose'])

    if args['plot']['soft']:
        path = args['output_dir'] / 'clustering'
        plot_soft_predictions(data=data,
                              contracted_tree=contracted_tree,
                              eq_cuts=cuts.equations,
                              path=path)

    ys_predicted = compute_hard_preditions(contracted_tree,
                                           cuts=cuts)

    if args['plot']['hard']:
        path = args['output_dir'] / 'clustering'
        plot_hard_predictions(data, ys_predicted, path=path)

    if data.ys is not None:
        compute_and_save_evaluation(ys=data.ys,
                                    ys_predicted=ys_predicted,
                                    hyperparameters=hyperparameters,
                                    id_run=id_run,
                                    path=args['output_dir'])


if __name__ == '__main__':

    # Make parser and parse command line
    #parser = make_parser()
    #args_parser = parser.parse_args()
    #args_parser = load_validate_parser(args_parser)

    root_dir = Path(__file__).resolve().parent
    cfg_file_path = root_dir / 'settings.yml'
    args_cfg_file = load_validate_config_file(cfg_file_path)

    args = args_cfg_file

    args = deactivate_plots(args)
    args = set_up_dirs(args, root_dir=root_dir)

    main(args)
