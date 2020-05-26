import sys
sys.path.append("..")

import argparse
from itertools import product

import pandas as pd

from sklearn.cluster import SpectralClustering, AffinityPropagation, KMeans, AgglomerativeClustering
from sklearn.metrics import homogeneity_score, adjusted_rand_score

from src.datasets import load_SBM

def benchmark_SMB(path, seed):

    results = pd.DataFrame(columns=['p', 'q', 'score', 'method'])

    ps = [0.05, 0.09, 0.14, 0.18, 0.23, 0.28, 0.32, 0.37, 0.41, 0.46, 0.51, 0.55, 0.6, 0.64, 0.69, 0.74, 0.78, 0.83, 0.87, 0.92]
    qs = [0.05, 0.09, 0.14, 0.18, 0.23, 0.28, 0.32, 0.37, 0.41, 0.46, 0.51, 0.55, 0.6, 0.64, 0.69, 0.74, 0.78, 0.83, 0.87, 0.92]
    block_sizes = [100, 100]
    seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    
    for p, q in product(ps, qs):
        
        print(f'Working on (p={p}, q={q}) with seed = {seed}')
        A, ys, G = load_SBM(block_sizes=block_sizes, p_in=p, p_out=q, seed=seed)

        sc = SpectralClustering(n_clusters=2, affinity='precomputed', n_init=100, random_state=seed)
    
        algs = [(sc, 'Spectral Custering')]

        for alg, label in algs:

            ys_predict = alg.fit_predict(A) 
            score = adjusted_rand_score(labels_true=ys, labels_pred=ys_predict)
            result = {'p': p, 'q': q, 'score': score, 'method': label, 'seed': seed}
            results = results.append(result, ignore_index=True)

    results.to_csv(f'benchmark_SMB_SpecClus_{seed}.csv', index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Program to benckmars')
    parser.add_argument('-s', dest='seed', action='store', type=int)
    args = parser.parse_args()

    benchmark_SMB("./", seed=args.seed)