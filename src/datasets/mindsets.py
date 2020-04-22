import numpy as np
import pandas as pd


def make_mindsets(args):

    sizes = args['mindset_sizes']
    n = sum(sizes)
    m = args['questions']
    p = args['noise']
    p_q = args['noise_questions']

    mindsets = np.empty([len(sizes), m])
    ys = np.empty([n,])
    xs = np.empty([n, m])


    counter = 0
    for k_iter, size in enumerate(sizes):
        # create ground truth mindset
        mindsets[k_iter] = np.random.randint(2, size=m)
        for _ in range(size):
            # add noise
            ys[counter] = k_iter
            flip = np.random.rand(m)
            xs[counter] = np.where(flip <= p, mindsets[k_iter], np.logical_not(mindsets[k_iter]))

            counter += 1

    # add noise question like gender etc.
    if p_q is not None:
        xs[:, :p_q] = np.random.randint(2, size=[n, p_q])

    return xs, ys
