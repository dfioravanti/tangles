from src.utils import dict_product
import sys


sys.path.append("../src")

parameters = {}
multi_parameters = {}

parameters['-r'] = 10
parameters['--id'] = 15


multi_parameters['-o'] = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
multi_parameters['-a'] = [10, 20, 30, 40, 50, 60]

parameters['-t'] = 'sbm'
parameters['-b'] = 'FM'
parameters['--lb_f'] = 0.2
parameters['-c'] = 'cut_sum'
parameters['--sbm_sizes'] = [100, 100]
parameters['--sbm_p'] = 0.3
parameters['--sbm_q'] = 0.1
parameters['--prune'] = 0
parameters['--nb_cuts'] = 20
parameters['-s'] = 10


with open('parameters_a_psi_sbm.txt', 'w') as f:
    for current_parameters in dict_product(multi_parameters):

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
