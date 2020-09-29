import sys

sys.path.append("../src")

import datetime

import numpy as np

from src.utils import dict_product

parameters = {}
multi_parameters = {}

parameters['-r'] = 10
parameters['--id'] = 1

parameters['-t'] = 'gau_mix'
parameters['-b'] = 'rand_proj'
parameters['-c'] = 'euclidean_mean'
parameters['-o'] = 100
parameters['--nb_cuts'] = 20
multi_parameters['--gauss_sizes'] = [[10, 5, 5], [100, 50, 50], [1000, 500, 500], [10000, 5000, 5000], [100000, 50000, 50000]]
parameters['--gauss_mean'] = '"[[2,2],[2,-2],[-2,2]]"'
parameters['--gauss_var'] = '"[[1,1],[1,1],[1,1]]"'


with open('parameters.txt', 'w') as f:
    for current_parameters in dict_product(multi_parameters):

        parameters['-a'] = int(sum(current_parameters['--gauss_sizes']) / 6)

        p = {**parameters, **current_parameters}

        line = []

        for k, v in p.items():
            if type(v) == list:
                str_v = (' '.join(map(str, v)))
            else:
                str_v = str(v)

            line += [f'{k} {str_v}']

        line = ' '.join(line)
        print(line, file=f)


# parameters['-t'] = 'sbm'
# parameters['-b'] = 'KL'
# parameters['-c'] = 'cut_sum'
# multi_parameters['-o'] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# multi_parameters['-a'] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# multi_parameters['--nb_cuts'] = [20]
# multi_parameters['--lb_f'] = [0.0]
# multi_parameters['--sbm_sizes'] = [[100, 100]]
# multi_parameters['--sbm_p'] = [0.3]
# multi_parameters['--sbm_q'] = [0.1]
#
# with open('parameters.txt', 'a') as f:
#     for current_parameters in dict_product(multi_parameters):
#
#         p = {**parameters, **current_parameters}
#
#         line = []
#
#         for k, v in p.items():
#             if type(v) == list:
#                 str_v = (' '.join(map(str, v)))
#             else:
#                 str_v = str(v)
#
#             line += [f'{k} {str_v}']
#
#         line = ' '.join(line)
#         print(line, file=f)