from random import sample

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.metrics import homogeneity_completeness_v_measure

from src.datasets.graphs import load_POLI_BOOKS, load_RPG
from src.order_functions import cut_order
from src.preprocessing import neighbours_in_same_cluster, build_cover_graph


def clustering(A, ys, nb_neigh, max_k):

    nb_verteces, _ = A.shape
    ys_pred = np.zeros(nb_verteces)
    idx_to_cover = set(range(nb_verteces))
    cover = []
    cuts = []

    while len(idx_to_cover) > 0:
        vertex = sample(idx_to_cover, 1)
        cut = neighbours_in_same_cluster(vertex, A, nb_neigh)
        cuts.append(cut)
        blob = set(np.where(cut == True)[0])
        cover.append(blob)
        idx_to_cover = idx_to_cover - blob

    initial_cuts = np.stack(cuts, axis=0)
    cuts = []
    A_cover = build_cover_graph(cover, A)
    hom = []
    cmp = []

    for i, k in enumerate(range(2, max_k), 1):
        lol = []
        cls = SpectralClustering(n_clusters=k, affinity='precomputed')
        clusters = cls.fit_predict(X=A_cover)

        for cluster in range(0, k):
            cuts_in_cluster = initial_cuts[clusters == cluster]
            cut = np.any(cuts_in_cluster, axis=0)
            if np.any(cut) and not np.all(cut):
                lol.append(cut_order(A, cut))

        print(np.array(lol).mean())

        for blob, cluster in zip(cover, clusters):
            ys_pred[list(blob)] = cluster
        homogeneity, completeness, _ = homogeneity_completeness_v_measure(ys, ys_pred)
        hom.append(homogeneity)
        cmp.append(completeness)

    return hom, cmp

fig, (ax1, ax2) = plt.subplots(1, 2)

A, ys = load_RPG(block_size=100, nb_blocks=2, p_in=.9, p_out=.2)
max_k = 6
ks = np.arange(2, max_k)
for nb_neigh in range(1, 10):
    hom, cmp = clustering(A, ys, nb_neigh, max_k)
    ax1.scatter(ks, hom, label=f'{nb_neigh}')
    ax2.scatter(ks, cmp)

    print(hom)
    print(cmp)

ax1.legend()
ax2.legend()
fig.show()
