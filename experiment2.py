from pathlib import Path

from src.parser import make_parser
from src.config import load_validate_settings, set_up_dirs
from src.plotting_experiments import quality_of_initial_cuts, comparison_of_initial_cuts, choice_of_a, \
    choice_of_cost_function, number_of_cuts, influence_of_cut_quality, choice_of_psi, interplay_cut_quality_and_a, \
    interplay_cut_quality_and_psi, interplay_a_and_psi, plot_ideal_cuts_distribution, plot_distributions, \
    influence_of_a, plot_choice_of_a


def main_tree(args):
    experiments = ['two', 'unbalanced', 'three']
    costfunctions = ['euclidean_sum']
    args['experiment']['dataset_name'] = 'knn_gauss_blobs'

    args['experiment']['agreement'] = 0
    args['experiment']['percentile_orders'] = 30
    args['experiment']['preprocessing_name'] = 'random_projection'

    args['preprocessing']['dimension'] = 1
    args['preprocessing']['nb_cuts'] = 50

    for costfun in costfunctions:
        print("Running experiments for: \t", costfun)
        args['experiment']['cost_function'] = costfun
        for name in experiments:
            if name == 'two':
                args['dataset']['blob_sizes'] = [500, 500]
                args['dataset']['blob_centers'] = [[-1.5, -1.5], [1.5, 1.5]]
                args['dataset']['blob_variances'] = [[1, 1], [1, 1]]
                args['dataset']['k'] = 10
            elif name == 'unbalanced':
                args['dataset']['blob_sizes'] = [700, 300]
                args['dataset']['blob_centers'] = [[-1.5, -1.5], [1.5, 1.5]]
                args['dataset']['blob_variances'] = [[1, 1], [1, 1]]
                args['dataset']['k'] = 10
            elif name == 'three':
                args['dataset']['blob_sizes'] = [330, 330, 330]
                args['dataset']['blob_centers'] = [[-1, -1], [1, 1], [1, -1]]
                args['dataset']['blob_variances'] = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
                args['dataset']['k'] = 10

            print("\nExperiment 2: influence of a given cost function\n")
            #plot_distributions(args, name)
            #plot_ideal_cuts_distribution(args)
            #influence_of_a(args, name)
            #comparison_of_initial_cuts(args, name)

    plot_choice_of_a()


if __name__ == '__main__':

    # Make parser, read inputs from command line and resolve paths

    parser = make_parser()
    args_parser = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    print(root_dir)
    args = load_validate_settings(args_parser, root_dir=root_dir)

    main_tree(args)
