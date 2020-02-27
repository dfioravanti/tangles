import numpy as np

from sklearn.metrics.pairwise import manhattan_distances


def order_questionnaire(xs, n_samples, cut):
    """
        idx = np.arange(len(xs))
        idx_in = np.random.choice(idx[cut], size=n_samples, replace=False)
        idx_out = np.random.choice(idx[~cut], size=n_samples, replace=False)

        in_cut = xs[idx_in, :]
        out_cut = xs[idx_out, :]
    """
    in_cut = xs[cut, :]
    out_cut = xs[~cut, :]

    orders = manhattan_distances(in_cut, out_cut)
    expected_order = np.average(orders)

    return expected_order





