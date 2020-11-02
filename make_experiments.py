import datetime

import numpy as np

from src.config import DATASET_SBM, DATASET_MINDSETS, PREPROCESSING_FID_MAT, PREPROCESSING_USE_FEATURES
from src.utils import dict_product

ts = int(datetime.datetime.now().timestamp())

parameters = {}
multi_parameters = {}

parameters['--id'] = ts
parameters['-t'] = DATASET_MINDSETS

multi_parameters['-s'] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
if parameters['-t'] == DATASET_SBM:

    sizes = np.linspace(5, 60, 10).astype(int)

    parameters['-p'] = PREPROCESSING_FID_MAT
    multi_parameters['-o'] = [100]
    multi_parameters['-a'] = [50]
    multi_parameters['--nb_cuts'] = [50]
    multi_parameters['--lb_f'] = [0.4]
    multi_parameters['--sbm_bs'] = [[100, 100]]
    multi_parameters['--sbm_p'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    multi_parameters['--sbm_q'] = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


elif parameters['-t'] == DATASET_MINDSETS:

    sizes = np.linspace(5, 55, 9).astype(int)

    parameters['-p'] = PREPROCESSING_USE_FEATURES
    multi_parameters['-o'] = [100]
    multi_parameters['--mind_sizes'] = [[size, size] for size in sizes]
    multi_parameters['--mind_questions'] = [40]
    multi_parameters['--mind_useless'] = [0]
    multi_parameters['--mind_noise'] = np.linspace(0, 0.24, 13)

with open('parameters.txt', 'w') as f:
    for current_parameters in dict_product(multi_parameters):

        if parameters['-t'] == DATASET_MINDSETS:
            parameters['-a'] = current_parameters['--mind_sizes'][0] // 3

        p = {**parameters, **current_parameters}

        line = []

        for k, v in p.items():
            if type(v) == list:
                str_v = (' '.join(map(str, v)))
            else:
                str_v = str(v)

            line += [f'{k} {str_v}']

        line += ['--no_plots']

        line = ' '.join(line)
        print(line, file=f)
