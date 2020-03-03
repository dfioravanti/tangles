import pathlib
from datetime import datetime
import itertools

import numpy as np
from scipy.linalg import hadamard


def make_centers(n_features=20, n_mindsets=2):

    output = np.zeros((n_mindsets, n_features), dtype=bool)

    k = np.int(np.ceil(np.log2(n_features)))
    n_blocks, len_final_block = divmod(n_features, k)

    block = np.array([list(i) for i in itertools.product([0, 1], repeat=k)])
    block = block[:n_mindsets]

    if len_final_block > 0:
        output[:, :-len_final_block] = np.tile(block, n_blocks)
        final_block = np.array([list(i) for i in itertools.product([0, 1], repeat=k)])
        final_block = final_block[:n_mindsets, :len_final_block]
        output[:, -len_final_block:] = final_block
    else:
        output = np.tile(block, n_blocks)

    return output


def make_centers2(n_features=20, n_mindsets=2):

    """
    Generates n_mindsets centers that are have hamming distance bigger than
    n_features // 2 - 2. This statement should be correct but I did not prove it.

    For now we generate those point by cropping a hadamarn matrix.
    TODO: Check out if we can use some other type of code that have a guaranteed hamming distance.
    TODO: It does not really scale up too much so better fix that to do soon

    Parameters
    ----------
    n_features : int, optional (default=20)
        The number of features.
    n_mindsets : int, optional (default=2)
        The number of classes, we call them "mindsets", that we should generate.

    Returns
    -------
    c : array of shape [n_mindsets, n_features]
        The coordinates of the centers of the mindsets

    """

    next_power_two = np.int(np.power(2, np.ceil(np.log2(n_features))))
    h_matrix = hadamard(next_power_two)
    h_matrix = (h_matrix == 1)

    idxs = np.arange(next_power_two)
    idxs = np.random.choice(idxs, n_mindsets, replace=False)
    c = h_matrix[idxs, :n_features]

    return c


def make_synthetic_questionnaire(n_samples=100, n_features=20, n_mindsets=2, tolerance=0.2,
                                 seed=None, centers=True):
    """

    This function simulates a synthetic questionnaire.

    The center of the mindsets will always disagree among each other in n_features / n_mindsets features.
    TODO: This is false, think how to fix it. I am not even sure it is possible to fix, it might be NP complete
          since it is equivalent to some reformulation of the clique problem.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of samples.
    n_features : int, optional (default=20)
        The number of features.
    n_mindsets : int, optional (default=2)
        The number of classes, we call them "mindsets", that we should generate.
    tolerance: float, optional (default=0.2)
        The percentage of deviation, computed respected to the 0-1 loss, that we allow
         inside a mindset. Must be in [0,1].
    seed: Int, optional (default=None)
        If provided it sets the seed of the random generator
    centers : bool, optional (default=False)
        If True the function will return an additional array that contains the
        coordinates of the center of the mindset.

    Returns
    -------

    xs : array of shape [n_samples, n_features]
        The generated samples.
    ys : array of shape [n_samples]
        The integer labels for mindset membership of each sample.
    cs : array of shape [n_mindsets, n_features]
        The coordinates of the centers of the mindsets

    """

    # TODO: We need some check on the number of mindsets vs number of samples and features.
    #       think how to implement that.

    if not 0 < tolerance < 1:
        raise ValueError("tolerance must be in [0,1]")

    if seed is not None:
        np.random.seed(seed)

    max_n_errors = np.floor(n_features * tolerance).astype(int)

    cs = make_centers(n_features, n_mindsets)
    xs, ys = np.empty((n_samples, n_features), dtype=np.bool), np.empty(n_samples, dtype=np.int)
    id_mindsets = np.array_split(np.arange(n_samples), n_mindsets)

    for (mindset, ids) in enumerate(id_mindsets):
        xs[ids, :] = cs[mindset]
        ys[ids] = mindset

    id_errors = np.random.choice(n_features, size=(n_samples, max_n_errors))
    xs_to_flip = xs[np.arange(len(id_errors)), id_errors.T]
    xs[np.arange(len(id_errors)), id_errors.T] = np.flip(xs_to_flip)

    if centers:
        return xs, ys, cs
    else:
        return xs, ys


if __name__ == '__main__':
    n_samples = 300
    n_features = 6
    n_mindsets = 3
    tolerance = 0.2
    seed = 42

    name = f'synthetic_s_{n_samples}_f_{n_features}_m_{n_mindsets}_t_{tolerance * 100:.1f}%'
    path = pathlib.Path('../../../datasets/questionnaire') / name
    path.mkdir(parents=True, exist_ok=True)

    README = f'Informations for the dataset: {name} \n \n' \
             f'samples: {n_samples} \n' \
             f'features: {n_features} \n' \
             f'mindsets: {n_mindsets} \n' \
             f'tolerance: {tolerance} \n' \
             f'seed: {seed} \n \n' \
             f'created on {datetime.now().isoformat()} \n'

    xs, ys, cs = make_synthetic_questionnaire(n_samples, n_features, n_mindsets, tolerance, seed)

    np.savetxt(path / "xs.txt", xs, fmt="%d")
    np.savetxt(path / "ys.txt", ys, fmt="%d")
    np.savetxt(path / "cs.txt", cs, fmt="%d")

    with open(path / "README", "w+") as f:
        f.write(README)
