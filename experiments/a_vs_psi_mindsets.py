from src.utils import dict_product
import sys


sys.path.append("../src")

parameters = {}
multi_parameters = {}

parameters['-r'] = 10
parameters['--id'] = 16

parameters['-t'] = 'mind'
parameters['-b'] = 'fea'
parameters['-c'] = 'manhattan_sum'
multi_parameters['-o'] = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
multi_parameters['-a'] = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
parameters['--prune'] = 0
parameters['-s'] = 10

parameters['--mind_sizes'] = [100, 50, 50]
parameters['--mind_questions'] = 10
parameters['--mind_useless'] = 5
parameters['--mind_noise'] = 0.1


with open('parameters_a_psi_mindset.txt', 'w') as f:
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
