from pathlib import Path

from src.parser import make_parser
from src.config import load_validate_settings, set_up_dirs
from src.plotting_experiments import quality_of_initial_cuts, comparison_of_initial_cuts, choice_of_a, \
    choice_of_cost_function, number_of_cuts, influence_of_cut_quality, choice_of_psi, interplay_cut_quality_and_a, \
    interplay_cut_quality_and_psi, interplay_a_and_psi


def main_tree(args):
    experiments = ['two', 'unbalanced', 'three']
    datasets = ['knn_gauss_blobs', 'sbm']

    for dataset in datasets:
        args['experiment']['dataset_name'] = dataset
        args['experiment']['agreement'] = 10
        args['experiment']['percentile_orders'] = 100
        print("Running experiments for: \t", dataset)
        for name in experiments:
            print("Cluster Settings: \t", name)
            if dataset == 'sbm':
                args['dataset']['p'] = 0.3
                args['dataset']['q'] = 0.1
                args['experiment']['preprocessing_name'] = 'fid_mat'
                args['preprocessing']['lb_f'] = 0.2
                args['experiment']['cost_function'] = 'cut'
                if name == 'two':
                    args['dataset']['block_sizes'] = [50, 50]
                elif name == 'unbalanced':
                    args['dataset']['block_sizes'] = [70, 30]
                elif name == 'three':
                    args['dataset']['block_sizes'] = [33, 33, 33]
            elif dataset == 'knn_gauss_blobs':
                args['experiment']['preprocessing_name'] = 'random_projection'
                args['experiment']['cost_function'] = 'euclidean'
                if name == 'two':
                    args['dataset']['blob_sizes'] = [50, 50]
                    args['dataset']['blob_centers'] = [[-2, -2], [2, 2]]
                    args['dataset']['blob_variances'] = [[1, 1], [1, 1]]
                    args['dataset']['k'] = 10
                elif name == 'unbalanced':
                    args['dataset']['blob_sizes'] = [70, 30]
                    args['dataset']['blob_centers'] = [[-2, -2], [2, 2]]
                    args['dataset']['blob_variances'] = [[1, 1], [1, 1]]
                    args['dataset']['k'] = 10
                elif name == 'three':
                    args['dataset']['blob_sizes'] = [33, 33, 33]
                    args['dataset']['blob_centers'] = [[-2, -2], [2, 2], [2, -2]]
                    args['dataset']['blob_variances'] = [[1, 1], [1, 1], [1, 1]]
                    args['dataset']['k'] = 10


            args['preprocessing']['nb_cuts'] = 20

            print("\nExperiment 9: Interplay - cut quality and psi\n")
            interplay_cut_quality_and_psi(args, name)


if __name__ == '__main__':

    # Make parser, read inputs from command line and resolve paths

    parser = make_parser()
    args_parser = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    print(root_dir)
    args = load_validate_settings(args_parser, root_dir=root_dir)

    main_tree(args)
