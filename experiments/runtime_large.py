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
parameters['--nb_cuts'] = 40
multi_parameters['--gauss_sizes'] = [[15, 15, 15, 35, 20],
                                     [150, 150, 150, 350, 200],
                                     [1500, 1500, 1500, 3500, 2000],
                                     [15000, 15000, 15000, 35000, 20000],
                                     [150000, 150000, 150000, 350000, 200000],
                                     [1500000, 1500000, 1500000, 3500000, 2000000],
                                     [15000000, 15000000, 15000000, 35000000, 20000000]]

parameters['--gauss_mean'] = '"[[-1,-7],[-1,0],[-1,7],[8,5],[8,-6]]"'
parameters['--gauss_var'] = '"[[2,2],[2.5,1.2],[2,2],[1.6,3],[1.5,2]]"'
parameters['--sample_cost'] = 100


with open('parameters.txt', 'w') as f:
    for current_parameters in dict_product(multi_parameters):

        parameters['-a'] = 8 * int(sum(current_parameters['--gauss_sizes']) / 100)

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
