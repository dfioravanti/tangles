import sys
sys.path.append("..")

import argparse

import pandas as pd

from sklearn.cluster import SpectralClustering, KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

from src.loading import get_dataset_and_order_function

from src.config import DATASET_SBM, DATASET_QUESTIONNAIRE, DATASET_BLOBS, DATASET_CANCER, DATASET_CANCER10, \
    DATASET_MINDSETS, DATASET_RETINAL, DATASET_MIES, DATASET_MOONS

def benchmarks(path, seed):
    
    args = {}
    args['experiment'] = {}
    args['dataset'] = {}    
        
    results = pd.DataFrame(columns=['dataset_name', 'Adjusted Rand Score', 'method', 'seed'])
    
    sc2 = SpectralClustering(n_clusters=2, random_state=seed)
    km2 = KMeans(n_clusters=2, random_state=seed)
    w = AgglomerativeClustering(n_clusters=2)
    
    algs = [(sc2, 'Spectral Custering'),
            (km2, 'KMeans'),
            (w, 'Ward')]
    
    args['experiment']['dataset_name'] = DATASET_CANCER
    args['dataset']['nb_bins'] = 0
    data_cancer, _ = get_dataset_and_order_function(args=args)
    
    for alg, label in algs:
        
        if alg != sc2:

            ys_predict = alg.fit_predict(data_cancer['xs']) 
            score = adjusted_rand_score(labels_true=data_cancer['ys'], labels_pred=ys_predict)
            result = {'dataset_name': DATASET_CANCER, 'Adjusted Rand Score': score, 'method': label, 'seed': seed}
            results = results.append(result, ignore_index=True)
    
    args['experiment']['dataset_name'] = DATASET_MOONS
    args['dataset']['n_samples'] = 300
    args['dataset']['noise'] = 0.1
    args['dataset']['radius'] = 0.1
    args['experiment']['seed'] = seed
    data_low_noise_moons, _ = get_dataset_and_order_function(args=args)
    
    for alg, label in algs:
        
        ys_predict = alg.fit_predict(data_low_noise_moons['xs']) 
        score = adjusted_rand_score(labels_true=data_low_noise_moons['ys'], labels_pred=ys_predict)
        result = {'dataset_name': f'{DATASET_MOONS}_low_nose', 'Adjusted Rand Score': score, 'method': label, 'seed': seed}
        results = results.append(result, ignore_index=True)

    args['experiment']['dataset_name'] = DATASET_MOONS
    args['dataset']['n_samples'] = 300
    args['dataset']['noise'] = 0.2
    args['dataset']['radius'] = 0.1
    args['experiment']['seed'] = seed
    data_high_noise_moons, _ = get_dataset_and_order_function(args=args)
    
    for alg, label in algs:
        
        ys_predict = alg.fit_predict(data_high_noise_moons['xs']) 
        score = adjusted_rand_score(labels_true=data_high_noise_moons['ys'], labels_pred=ys_predict)
        result = {'dataset_name': f'{DATASET_MOONS}_high_nose', 'Adjusted Rand Score': score, 'method': label, 'seed': seed}
        results = results.append(result, ignore_index=True)
        
    args['experiment']['dataset_name'] = DATASET_BLOBS
    args['dataset']['blob_sizes'] = [60, 60]
    args['dataset']['blob_centers'] = [[-2, -2], [2, 2]]
    args['dataset']['sigma'] = 1
    args['dataset']['radius'] = 0.3
    args['experiment']['seed'] = seed
    data_low_noise_blobs, _ = get_dataset_and_order_function(args=args)
    
    for alg, label in algs:
        
        ys_predict = alg.fit_predict(data_low_noise_blobs['xs']) 
        score = adjusted_rand_score(labels_true=data_low_noise_blobs['ys'], labels_pred=ys_predict)
        result = {'dataset_name': f'{DATASET_BLOBS}_2_low_nose', 'Adjusted Rand Score': score, 'method': label, 'seed': seed}
        results = results.append(result, ignore_index=True)
        
    args['experiment']['dataset_name'] = DATASET_BLOBS
    args['dataset']['blob_sizes'] = [60, 60]
    args['dataset']['blob_centers'] = [[-2, -2], [2, 2]]
    args['dataset']['sigma'] = 1.5
    args['dataset']['radius'] = 0.3
    args['experiment']['seed'] = seed
    data_low_noise_blobs, _ = get_dataset_and_order_function(args=args)
    
    for alg, label in algs:
        
        ys_predict = alg.fit_predict(data_low_noise_blobs['xs']) 
        score = adjusted_rand_score(labels_true=data_low_noise_blobs['ys'], labels_pred=ys_predict)
        result = {'dataset_name': f'{DATASET_BLOBS}_2_high_nose', 'Adjusted Rand Score': score, 'method': label, 'seed': seed}
        results = results.append(result, ignore_index=True)
            
    km4 = KMeans(n_clusters=4, random_state=seed)
    sc4 = SpectralClustering(n_clusters=4, random_state=seed)
    
    algs = [(km4, 'Spectral Custering'),
            (sc4, 'KMeans'),
            (w, 'Ward')]
    
    args['experiment']['dataset_name'] = DATASET_BLOBS
    args['dataset']['blob_sizes'] = [60, 60, 60, 60]
    args['dataset']['blob_centers'] = [[-2, 2], [-2, -2], [2, -2], [2, 2]]
    args['dataset']['sigma'] = 1
    args['dataset']['radius'] = 0.3
    args['experiment']['seed'] = seed
    data_low_noise_blobs, _ = get_dataset_and_order_function(args=args)
    
    for alg, label in algs:
        
        ys_predict = alg.fit_predict(data_low_noise_blobs['xs']) 
        score = adjusted_rand_score(labels_true=data_low_noise_blobs['ys'], labels_pred=ys_predict)
        result = {'dataset_name': f'{DATASET_BLOBS}_4_low_nose', 'Adjusted Rand Score': score, 'method': label, 'seed': seed}
        results = results.append(result, ignore_index=True)

    args['experiment']['dataset_name'] = DATASET_BLOBS
    args['dataset']['blob_sizes'] = [60, 60, 60, 60]
    args['dataset']['blob_centers'] = [[-2, 2], [-2, -2], [2, -2], [2, 2]]
    args['dataset']['sigma'] = 1.5
    args['dataset']['radius'] = 0.3
    args['experiment']['seed'] = seed
    data_high_noise_4_blobs, _ = get_dataset_and_order_function(args=args)
    
    for alg, label in algs:
        
        ys_predict = alg.fit_predict(data_high_noise_4_blobs['xs']) 
        score = adjusted_rand_score(labels_true=data_high_noise_4_blobs['ys'], labels_pred=ys_predict)
        result = {'dataset_name': f'{DATASET_BLOBS}_4_high_nose', 'Adjusted Rand Score': score, 'method': label, 'seed': seed}
        results = results.append(result, ignore_index=True)

    results.to_csv(f'{path}/benchmarks_{seed}.csv', index=False)             
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Program to benckmars')
    parser.add_argument('-s', dest='seed', action='store', type=int, default=42)
    args = parser.parse_args()

    benchmarks("./", seed=args.seed)
        