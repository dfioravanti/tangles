import numpy as np
import pandas as pd

from sklearn.neighbors import kneighbors_graph
from sklearn.datasets import load_breast_cancer, make_blobs

import networkx as nx


def load_BIG5(path):

    df = pd.read_csv(path, header=0, sep='\t')
    df = df.drop(['race', 'age', 'engnat', 'gender', 'hand', 'source', 'country'], axis=1)

    xs = df.to_numpy()
    idx = np.random.choice(len(xs), size=300, replace=False)
    xs = xs[idx]
    ys = None

    return xs, ys


def load_CANCER():

    data = load_breast_cancer()
    feature_names, xs, ys = data.feature_names, data.data, data.target

    return xs, ys


def load_CANCER10(path):

    df = pd.read_csv(path, header=0)
    name_columns = df.columns.values

    ys = np.array(df["class"])
    xs = df[name_columns[:-1]]

    xs = xs.to_numpy()

    return xs, ys


def load_LFR(nb_nodes, tau1, tau2, mu, min_community, average_degree, seed):

    A = np.zeros((nb_nodes, nb_nodes), dtype=bool)
    ys = np.zeros(nb_nodes, dtype=int)
    G = nx.LFR_benchmark_graph(nb_nodes, tau1, tau2, mu,
                                   min_community=min_community, average_degree=average_degree,
                                   seed=seed)

    for node, ad in G.adjacency():
        A[node, list(ad.keys())] = True

    partitions = {frozenset(G.nodes[v]['community']) for v in G}
    for cls, points in enumerate(partitions):
        ys[list(points)] = cls

    return A, ys, G


def load_GMM(blob_sizes, blob_centers, blob_variances, seed):

    xs, ys = make_blobs(n_samples=blob_sizes, centers=blob_centers, cluster_std=blob_variances, n_features=2, random_state=seed)

    return xs, ys


def load_SBM(block_sizes, p_in, p_out, seed):

    nb_nodes = np.sum(block_sizes)

    A = np.zeros((nb_nodes, nb_nodes), dtype=bool)
    ys = np.zeros(nb_nodes, dtype=int)
    G = nx.random_partition_graph(block_sizes, p_in, p_out, seed=seed)

    for node, ad in G.adjacency():
        A[node, list(ad.keys())] = True

    for cls, points in enumerate(G.graph["partition"]):
        ys[list(points)] = cls

    return A, ys, G


def load_MIES(root_path):

    path = root_path / 'datasets/mies/data.csv'

    df = pd.read_csv(path, sep='\t')

    # drop users that somehow did no answered to intro, extro, no
    df = df[df['IE'] != 0]

    answers = df.filter(regex=r'Q\d+A')
    labels = df['IE']

    xs = answers.to_numpy()
    ys = labels.to_numpy()

    return xs, ys


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

        xs.append(xs_mindset)
        ys.append(ys_mindset)

    xs = np.vstack(xs)
    ys = np.concatenate(ys)

    # Add noise
    noise_per_question = np.random.rand(nb_points, nb_questions)
    flip_question = noise_per_question < noise
    xs[flip_question] = np.logical_not(xs[flip_question])

    # add noise question like gender etc.
    if nb_useless is not None:
        mindsets = np.hstack((mindsets, np.full([nb_mindsets, nb_useless], 0.5)))
        useless = np.random.randint(2, size=[nb_points, nb_useless])
        xs = np.hstack((xs, useless))

    return xs, ys, mindsets


def make_likert_questionnaire(nb_samples, nb_features, nb_mindsets, centers, range_answers, seed=None):

    if seed is not None:
        np.random.seed(seed)

    min_answer = range_answers[0]
    max_answer = range_answers[1]

    xs = np.zeros((nb_samples, nb_features), dtype=int)
    ys = np.zeros(nb_samples, dtype=int)

    idxs = np.array_split(np.arange(nb_samples), nb_mindsets)

    if not centers:
        centers = np.random.random_integers(low=min_answer, high=max_answer, size=(nb_mindsets, nb_features))
    else:
        raise NotImplementedError

    for i in np.arange(nb_mindsets):

        nb_points = len(idxs[i])
        answers_mindset = np.random.normal(loc=centers[i], size=(nb_points, nb_features))
        answers_mindset = np.rint(answers_mindset)
        answers_mindset[answers_mindset > max_answer] = max_answer
        answers_mindset[answers_mindset < min_answer] = min_answer

        xs[idxs[i]] = answers_mindset
        ys[idxs[i]] = i

    return xs, ys, centers


def load_RETINAL(root_path, nb_bins, max_idx):

    path_xs = root_path / "datasets/retinal/rgc_data_X.npy"
    xs = np.load(path_xs).T

    if nb_bins != False:
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