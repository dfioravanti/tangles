import datetime

import numpy as np

from src.config import DATASET_SBM, DATASET_MINDSETS, PREPROCESSING_KARNIG_LIN, \
                       PREPROCESSING_FID_MAT, PREPROCESSING_USE_FEATURES, \
                       DATASET_QUESTIONNAIRE, PREPROCESSING_BINARIZED_LIKERT, PREPROCESSING_USE_FEATURES
from src.utils import dict_product

ts = int(datetime.datetime.now().timestamp())

parameters = {}
multi_parameters = {}

parameters['--id'] = ts
parameters['-t'] = DATASET_MINDSETS

multi_parameters['-s'] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#multi_parameters['-s'] = [42]
if parameters['-t'] == DATASET_SBM:

    """ parameters['-p'] = PREPROCESSING_FID_MAT
    multi_parameters['-o'] = [80]
    multi_parameters['--nb_cuts'] = [20]
    multi_parameters['--lb_f'] = [0.2, 0.3, 0.4]
    multi_parameters['-a'] = [50]
    multi_parameters['--sbm_bs'] = [[100, 100]]
    multi_parameters['--sbm_p'] = [0.05, 0.09, 0.14, 0.18, 0.23, 0.28, 0.32, 0.37, 0.41, 0.46, 0.51, 0.55, 0.6, 0.64, 0.69, 0.74, 0.78, 0.83, 0.87, 0.92]
    multi_parameters['--sbm_q'] = [0.05, 0.09, 0.14, 0.18, 0.23, 0.28, 0.32, 0.37, 0.41, 0.46, 0.51, 0.55, 0.6, 0.64, 0.69, 0.74, 0.78, 0.83, 0.87, 0.92]
 """
 
    sizes = np.linspace(5, 60, 10).astype(int)
 
    parameters['-p'] = PREPROCESSING_FID_MAT
    multi_parameters['-o'] = np.arange(1, 11) * 10
    multi_parameters['-a'] = sizes
    multi_parameters['--nb_cuts'] = [20]
    multi_parameters['--lb_f'] = [0.2]
    multi_parameters['--sbm_bs'] = [[size, size] for size in sizes]
    multi_parameters['--sbm_p'] = [0.3, 0.6]
    multi_parameters['--sbm_q'] = [0.1, 0.4]
    
 
elif parameters['-t'] == DATASET_MINDSETS:
        
    sizes = np.linspace(5, 60, 10).astype(int)
        
    parameters['-p'] = PREPROCESSING_USE_FEATURES
    multi_parameters['-o'] = [100]
    multi_parameters['-a'] = sizes
    multi_parameters['--mind_sizes'] = [[size, size] for size in sizes]
    multi_parameters['--mind_questions'] = [40]
    multi_parameters['--mind_useless'] = [0]
    multi_parameters['--mind_noise'] = [0.2]
    
with open('parameters.txt', 'w') as f:
    for current_parameters in dict_product(multi_parameters):
        
        if parameters['-t'] == DATASET_MINDSETS:
            parameters['-a'] = current_parameters['--mind_sizes'][1] // 3
        if parameters['-t'] == DATASET_SBM:
            parameters['-a'] = current_parameters['--sbm_bs'][1] // 2
                    
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
