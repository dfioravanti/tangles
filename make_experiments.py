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
    multi_parameters['-o'] = np.arange(1, 11) * 10
    multi_parameters['-a'] = np.linspace(10, 60, 11).astype(int)
    multi_parameters['--nb_cuts'] = [20]
    multi_parameters['--lb_f'] = [0.2]
    multi_parameters['--sbm_bs'] = [[100, 100]]
    multi_parameters['--sbm_p'] = [0.3]
    multi_parameters['--sbm_q'] = [0.1]


elif parameters['-t'] == DATASET_MINDSETS:

    sizes = np.linspace(5, 60, 10).astype(int)

    parameters['-p'] = PREPROCESSING_USE_FEATURES
    multi_parameters['-o'] = np.arange(1, 11) * 10
    multi_parameters['-a'] = np.linspace(10, 100, 9).astype(int)
    multi_parameters['--mind_sizes'] = [[40, 40]]
    multi_parameters['--mind_questions'] = [10, 20, 40]
    multi_parameters['--mind_useless'] = [10, 20, 40]
    multi_parameters['--mind_noise'] = [0.1, 0.2]

with open('parameters.txt', 'w') as f:
    for current_parameters in dict_product(multi_parameters):

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
