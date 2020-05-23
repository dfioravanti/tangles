import datetime

import numpy as np

from src.config import DATASET_SBM, DATASET_KNN_BLOBS, DATASET_MINDSETS, PREPROCESSING_KARNIG_LIN, \
                       PREPROCESSING_FID_MAT, PREPROCESSING_USE_FEATURES, DATASET_QUESTIONNAIRE, PREPROCESSING_BINARIZED_LIKERT
from src.utils import dict_product

ts = int(datetime.datetime.now().timestamp())

parameters = {}
multi_parameters = {}

parameters['-t'] = DATASET_SBM
parameters['--id'] = ts
parameters['-p'] = PREPROCESSING_KARNIG_LIN

#multi_parameters['-p'] = [PREPROCESSING_KARNIG_LIN, PREPROCESSING_FID_MAT]
multi_parameters['--nb_cuts'] = [200]
multi_parameters['--lb_f'] = [0.3, 0.4]

multi_parameters['-s'] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#multi_parameters['-s'] = [42]



if parameters['-t'] == DATASET_MINDSETS:
    
    multi_parameters['--mind_sizes'] = [[100, 100], [100, 100, 100], [100, 100, 100, 100], [100, 100, 100, 100, 100]]
    multi_parameters['--mind_sizes'] += [[50, 100], [60, 100], [70, 100], [80, 100], [90, 100]]

    multi_parameters['--mind_questions'] = np.arange(1, 11) * 10
    multi_parameters['--mind_useless'] = np.arange(1, 11) * 10

    multi_parameters['--mind_noise'] = [0.1]


elif parameters['-t'] == DATASET_SBM:

    multi_parameters['--sbm_bs'] = [[100, 100]]

    multi_parameters['--sbm_p'] = [0.05, 0.18, 0.28, 0.87, 0.64, 0.55, 0.51, 0.23, 0.37, 0.83, 0.46,
                                   0.32, 0.09, 0.92, 0.6 , 0.14, 0.74, 0.41, 0.78, 0.69]
    multi_parameters['--sbm_q'] = [0.05, 0.18, 0.28, 0.87, 0.64, 0.55, 0.51, 0.23, 0.37, 0.83, 0.46,
                                   0.32, 0.09, 0.92, 0.6 , 0.14, 0.74, 0.41, 0.78, 0.69]

elif parameters['-t'] == DATASET_KNN_BLOBS:

    multi_parameters['--gauss_bs'] = [[100, 100]]
    multi_parameters['--gauss_cs'] = [[-3, -3, 3, 3], [-2, -2, 2, 2], [-1.5, -1.5, 1.5, 1.5], [-1, -1, 1, 1]]
    multi_parameters['--gauss_k'] = np.arange(8, 20)

elif parameters['-t'] == DATASET_QUESTIONNAIRE:

    multi_parameters['--q_nb_samples'] = [1000]
    multi_parameters['--q_nb_features'] = np.arange(1, 20) * 10
    multi_parameters['--q_nb_mindsets'] = np.arange(2, 20)
    multi_parameters['--q_range_answers'] = [[1, i] for i in range(2, 20)]

with open('parameters.txt', 'w') as f:
    for current_parameters in dict_product(multi_parameters):
        if parameters['-t'] == DATASET_QUESTIONNAIRE:
            parameters['-a'] = int(np.floor(current_parameters['--q_nb_samples'] / (current_parameters['--q_nb_mindsets'] + 2)))
        if parameters['-t'] == DATASET_SBM:
            parameters['-a'] = int(np.floor(min(current_parameters['--sbm_bs']) / 2))
        if parameters['-t'] == DATASET_KNN_BLOBS:
            parameters['-a'] = int(np.floor(min(current_parameters['--gauss_bs']) / 3))

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
