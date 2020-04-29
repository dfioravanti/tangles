import numpy as np

def make_mindsets(mindset_sizes, nb_questions, nb_useless, noise, seed):

    if seed is not None:
        np.random.seed(seed)

    nb_points = sum(mindset_sizes)
    nb_mindsets = len(mindset_sizes)

    xs, ys = [], []

    # create ground truth mindset
    mindsets = np.random.randint(2, size=(nb_mindsets, nb_questions))

    for idx_mindset, size_mindset in enumerate(mindset_sizes):

        # Points without noise
        xs_mindset = np.tile(mindsets[idx_mindset], (size_mindset, 1))
        ys_mindset = np.repeat(idx_mindset, repeats=size_mindset, axis=0)

        # Add noise
        noise_per_question = np.random.rand(size_mindset, nb_questions)
        flip_question = noise_per_question <= noise
        xs_mindset[flip_question] = np.logical_not(xs_mindset[flip_question])

        xs.append(xs_mindset)
        ys.append(ys_mindset)

    xs = np.vstack(xs)
    ys = np.concatenate(ys)

    # add noise question like gender etc.
    if nb_useless is not None:
        useless = np.random.randint(2, size=[nb_points, nb_useless])
        xs = np.hstack((xs, useless))

    return xs, ys, mindsets
