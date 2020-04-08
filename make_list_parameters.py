from src.config import DATASET_SBM, DATASET_KNN_BLOBS, PREPROCESSING_KARNIG_LIN
from src.utils import dict_product

parameters = {}
multi_parameters = {}
parameters['-t'] = DATASET_KNN_BLOBS
parameters['-p'] = PREPROCESSING_KARNIG_LIN

#multi_parameters['-s'] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
multi_parameters['-s'] = [42]
multi_parameters['--KL_nb'] = [100, 200, 300]

if parameters['-t'] == DATASET_SBM:

    multi_parameters['--sbm_bs'] = [[100, 100], [100, 100, 100], [100, 100, 100, 100], [100, 100, 100, 100, 100]]

    multi_parameters['--sbm_bs'] += [[50, 100], [60, 100], [70, 100],
                                     [80, 100], [90, 100]]

    multi_parameters['--sbm_ps'] = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    multi_parameters['--sbm_qs'] = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

elif parameters['-t'] == DATASET_KNN_BLOBS:

    multi_parameters['--gauss_bs'] = [[100, 100]]

    multi_parameters['--gauss_cs'] = [[-3, -3, 3, 3], [-2, -2, 2, 2], [-1.5, -1.5, 1.5, 1.5], [-1, -1, 1, 1]]

    multi_parameters['--gauss_ks'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

with open('parameters.txt', 'w') as f:
    for current_parameters in dict_product(multi_parameters):
        if parameters['-t'] == DATASET_SBM:
            parameters['--KL_frac'] = list(range(2, len(current_parameters['--sbm_bs']) + 3))
        elif parameters['-t'] == DATASET_KNN_BLOBS:
            parameters['--KL_frac'] = list(range(2, len(current_parameters['--gauss_bs']) + 3))
        p = {**parameters, **current_parameters}

        line = []

        for k, v in p.items():
            if type(v) == list:
                str_v = (' '.join(map(str, v)))
            else:
                str_v = str(v)

            line += [f'{k} {str_v}']

        line += ['--no_plot_tangles', '--no_plot_cuts']

        line = ' '.join(line)
        print(line, file=f)
