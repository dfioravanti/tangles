import numpy as np
import pandas as pd

def load_RETINAL(root_path, nb_bins, max_idx):

    path_xs = root_path / "datasets/retinal/rgc_data_X.npy"
    xs = np.load(path_xs).T
    xs_df = pd.DataFrame(xs)
    xs_df = xs_df.apply(lambda x: pd.qcut(x, q=nb_bins, labels=list(range(1, nb_bins+1))), axis=0)
    xs = xs_df.to_numpy()
    xs = xs.astype(int)
    
    path_ys = root_path / "datasets/retinal/rgc_data_ci.npy"  
    ys = np.load(path_ys)
    
    idx_to_take = (ys <= max_idx)

    xs = xs[idx_to_take]
    ys = ys[idx_to_take]

    return xs, ys
