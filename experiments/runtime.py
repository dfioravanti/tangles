from src.utils import dict_product
import sys


sys.path.append("../src")

parameters = {}
multi_parameters = {}

parameters['-r'] = 10
parameters['--id'] = 1

parameters['-t'] = 'gau_mix'
parameters['-b'] = 'rand_proj'
parameters['-c'] = 'euclidean_mean'
parameters['-o'] = 100
parameters['--prune'] = 1
parameters['--nb_cuts'] = 25
parameters['-s'] = 10
multi_parameters['--gauss_sizes'] = [[20, 10, 20, 10],
                                     [200, 100, 200, 100],
                                     [2000, 1000, 2000, 1000],
                                     [20000, 10000, 20000, 10000],
                                     [200000, 100000, 200000, 100000],
                                     [2000000, 1000000, 2000000, 1000000]]

parameters['--gauss_mean'] = '"[[-3,-2],[-3,3],[3,3],[3,-2]]"'
parameters['--gauss_var'] = '"[[1,2],[1,0.8],[1,2],[1,0.8]]"'
parameters['--sample_cost'] = 100


with open('parameters.txt', 'w') as f:
    for current_parameters in dict_product(multi_parameters):

        parameters['-a'] = int( 6 * current_parameters['--gauss_sizes'][1] / 10)

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
