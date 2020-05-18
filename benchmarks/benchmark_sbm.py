import sys
sys.path.append("..")

from itertools import product

import pandas as pd

from sklearn.cluster import SpectralClustering, AffinityPropagation, KMeans, AgglomerativeClustering
from sklearn.metrics import homogeneity_score, adjusted_rand_score

from src.datasets import load_SBM

def benchmark_SMB(path):

    results = pd.DataFrame(columns=['p', 'q', 'score', 'method'])

    ps = [0.05, 0.09, 0.14, 0.18, 0.23, 0.28, 0.32, 0.37, 0.41, 0.46, 0.51, 0.55, 0.6, 0.64, 0.69, 0.74, 0.78, 0.83, 0.87, 0.92]
    qs = [0.05, 0.09, 0.14, 0.18, 0.23, 0.28, 0.32, 0.37, 0.41, 0.46, 0.51, 0.55, 0.6, 0.64, 0.69, 0.74, 0.78, 0.83, 0.87, 0.92]
    block_sizes = [100, 100]

    for p, q in product(ps, qs):
        
        print(f'Working on (p={p}, q={q})')
        A, ys, G = load_SBM(block_sizes=block_sizes, p_in=p, p_out=q, seed=42)

        sc = SpectralClustering(n_clusters=2, affinity='precomputed', n_init=100, random_state=42)
        af = AffinityPropagation(affinity='precomputed', random_state=42)
        km = KMeans(n_clusters=2, random_state=42)
        acc = AgglomerativeClustering(n_clusters=2, affinity='precomputed', linkage='complete')
        aac = AgglomerativeClustering(n_clusters=2, affinity='precomputed', linkage='average')
        asc = AgglomerativeClustering(n_clusters=2, affinity='precomputed', linkage='single')

        algs = [(sc, 'Spectral Custering'),
                (af, 'Affinity Propagation'),
                (km, 'K-means'),
                (acc, 'Agglomerative Complete Clustering'),
                (aac, 'Agglomerative Average Clustering'),
                (asc, 'Agglomerative Single Clustering')]

        for alg, label in algs:

            print(f'\tWorking on {label}')
            ys_predict = alg.fit_predict(A) 
            score = adjusted_rand_score(labels_true=ys, labels_pred=ys_predict)
            result = {'p': p, 'q': q, 'score': score, 'method': label}
            results = results.append(result, ignore_index=True)

    results.to_csv('benchmark_SMB.csv', index=False)

if __name__ == "__main__":

    benchmark_SMB("./")