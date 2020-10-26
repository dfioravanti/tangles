import multiprocessing
import time
import numpy as np
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import SpectralClustering, AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.my_types import Dataset

MAX_TIME = 60 * 60


def spectral_sbm_worker(n_clusters, data, seed, result):
    start = time.time()
    result["predicted"] = SpectralClustering(n_clusters=n_clusters, assign_labels='kmeans', affinity='precomputed',
                                             random_state=seed).fit(data.A).labels_
    result['time'] = time.time() - start


def linkage_sbm_worker(n_clusters, data, seed, result):
    start = time.time()
    result["predicted"] = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed',
                                                  linkage='average').fit(data.A).labels_
    result['time'] = time.time() - start


def linkage_gauss_worker(n_clusters, data, seed, result):
    start = time.time()
    result["predicted"] = AgglomerativeClustering(n_clusters=n_clusters,
                                                  linkage='average').fit(data.xs).labels_
    result['time'] = time.time() - start


def kmeans_gauss_worker(n_clusters, data, seed, result):
    start = time.time()
    result["predicted"] = KMeans(n_clusters=n_clusters, random_state=seed).fit(data.xs).labels_
    result['time'] = time.time() - start


def kmeans_sbm_worker(n_clusters, data, seed, result):
    start = time.time()
    result["predicted"] = KMeans(n_clusters=n_clusters, random_state=seed).fit(data.A).labels_
    result['time'] = time.time() - start


def spectral_gauss_worker(n_clusters, data, seed, result):
    start = time.time()
    result["predicted"] = SpectralClustering(n_clusters=n_clusters, assign_labels='kmeans',
                                             affinity='nearest_neighbors',
                                             n_neighbors=int(2 * np.log(data.xs.shape[0])), random_state=seed).fit(
        data.xs).labels_
    result['time'] = time.time() - start


def divisive_sbm_worker(nb_clusters, data, seed, result):
    start = time.time()
    result["predicted"] = divisive_spectral_clustering(data.A, nb_clusters, dataset='sbm')
    result['time'] = time.time() - start


def divisive_gauss_worker(nb_clusters, data, seed, result):
    start = time.time()
    result["predicted"] = divisive_spectral_clustering(data.xs, nb_clusters)
    result['time'] = time.time() - start


def compute_and_save_comparison_no_sc(data, hyperparameters, id_run, path, r=1):
    results = {}


    _, results = compare(kmeans_gauss_worker, data, hyperparameters, results,
                                   name='kmeans')

    #_, results = compare(linkage_gauss_worker, data, hyperparameters, results,
    #                               name='Average Linkage')

    results = pd.DataFrame(results, index=[r])
    if os.path.isfile(str(path / 'comparison_{}.csv'.format(id_run))):
        results.to_csv(str(path / 'comparison_{}.csv'.format(id_run)), mode='a', header=False)
    else:
        print(results.to_csv(str(path / 'comparison_{}.csv'.format(id_run))))



def compute_and_save_comparison(data, hyperparameters, id_run, path, r=1):
    results = {}

    if hyperparameters['dataset'] == Dataset.SBM:

        field_names, results = compare(spectral_sbm_worker, data, hyperparameters, results, name='Spectral Clustering')

        field_names, results = compare(linkage_sbm_worker, data, hyperparameters, results, field_names,
                                       name='Average Linkage')

        field_names, results = compare(divisive_sbm_worker, data, hyperparameters, results, field_names,
                                       name='Divisive')

        field_names, results = compare(kmeans_sbm_worker, data, hyperparameters, results, field_names,
                                       name='kmeans')

    else:
        field_names, results = compare(kmeans_gauss_worker, data, hyperparameters, results, name='kMeans')

        field_names, results = compare(spectral_gauss_worker, data, hyperparameters, results, field_names,
                                       name='Spectral Clustering')

        field_names, results = compare(linkage_gauss_worker, data, hyperparameters, results, field_names,
                                       name='Average Linkage')

        field_names, results = compare(divisive_gauss_worker, data, hyperparameters, results, field_names,
                                       name='Divisive')


    results = pd.DataFrame(results, index=[r])

    if os.path.isfile(str(path / 'comparison_{}.csv'.format(id_run))):
        results.to_csv(str(path / 'comparison_{}.csv'.format(id_run)), mode='a', header=False)
    else:
        results.to_csv(str(path / 'comparison_{}.csv'.format(id_run)))


def compare(worker, data, hyperparameters, results, field_names=[""], name=""):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    p = multiprocessing.Process(target=worker, name="spectral_eval_sbm",
                                args=(len(np.unique(data.ys)), data, hyperparameters['seed'], return_dict))
    p.start()
    p.join(MAX_TIME)

    if p.is_alive():
        print("process still alive after {} seconds.... kill it".format(MAX_TIME))
        p.terminate()
        p.join()
        ARS = None
        sc_time = None
        NMI = None
        print("Spectral clustering took too long!")
    else:
        sc_time = return_dict['time']
        ARS = adjusted_rand_score(data.ys, return_dict['predicted'])
        NMI = normalized_mutual_info_score(data.ys, return_dict['predicted'])

    field_names.append('Runtime {}'.format(name))
    field_names.append('Adjusted Rand Score {}'.format(name))
    field_names.append('NMI {}'.format(name))

    results['Adjusted Rand Score {}'.format(name)] = ARS
    results['Runtime {}'.format(name)] = sc_time
    results['NMI {}'.format(name)] = NMI

    print('{} Adjusted Rand Score: {}'.format(name, ARS), flush=True)
    print('{} Normalized Mutual Information: {}'.format(name, NMI), flush=True)

    return field_names, results


def divisive_spectral_clustering(data, nb_classes, dataset=None):
    data = np.array(data, dtype=float)
    predicted = np.zeros(data.shape[0], dtype=int)

    unique, counts = np.unique(predicted, return_counts=True)
    nb_clusters = len(unique)
    if dataset == 'sbm':
        while nb_clusters < nb_classes:
            split_next = predicted == unique[np.argmax(counts)]
            xs = data[np.ix_(split_next, split_next)]
            labels = SpectralClustering(n_clusters=2, assign_labels='kmeans',
                                        affinity='precomputed').fit(xs).labels_
            predicted[split_next] += labels * nb_clusters
            unique, counts = np.unique(predicted, return_counts=True)
            nb_clusters = len(unique)
    else:
        while nb_clusters < nb_classes:
            split_next = predicted == unique[np.argmax(counts)]
            xs = data[split_next, :]
            labels = SpectralClustering(n_clusters=2, assign_labels='kmeans',
                                        affinity='nearest_neighbors', n_neighbors=int(2 * np.log(sum(split_next)))).fit(
                xs).labels_
            predicted[split_next] += labels * nb_clusters
            unique, counts = np.unique(predicted, return_counts=True)
            nb_clusters = len(unique)


    return predicted
