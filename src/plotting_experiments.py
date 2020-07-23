import multiprocessing
from copy import deepcopy
from functools import partial

import matplotlib.pyplot as plt
import matplotlib as mlp

from sklearn import metrics
import numpy as np
import scipy.stats as ss
from sklearn.metrics import v_measure_score

from src.execution import compute_cuts, order_cuts, pick_cuts_up_to_order
from src.loading import get_dataset_and_order_function, resolve_cost_function
from src.preprocessing import fid_mat_algorithm
from src.tangle_tree import TangleTreeModel

runs = 20

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

# parameter sets that should be same for all
agreement = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
psis = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
algorithms = ['fid_mat', 'random_projection']
cost_functions = ['euclidean', 'cut', 'euclidean_sum', 'cut_sum']
nb_cuts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
quality = [[0.05, 0.1, 0.2, 0.3], [0.3], [0.2], [0.1], [0.05]]
lb_f = np.arange(0.1, 0.5, 0.05)

mycolors = ['blue', 'red', 'lime', 'orange', 'cyan', 'green', 'magenta', 'chocolate', 'deepskyblue', 'purple']
mymarkers = ['.', '^', '*', 'x', '3', 's', 'd', '+', '<', 'h']


def exp(cost):
    return np.exp(-3*cost)


def sigmoid(cost):
    return 1 / (1 + np.exp(10 * (cost - 0.4)))


def plot_ideal_cuts_distribution(args):
    data, order_function = get_dataset_and_order_function(args)
    cuts = []
    for j in range(1, 51):
        for k in range(50, 100):
            cut = np.zeros(100, dtype=bool)
            cut[0:j] = True
            cut[50:k] = True
            cut_cost = order_function(cut)
            cut_homo = max(metrics.homogeneity_score(data['ys'], cut), metrics.homogeneity_score(cut, data['ys']))
            cuts.append([cut_cost, cut_homo])

    cuts = np.array(cuts)

    data_cost = cuts[:, 0]
    data_homo = cuts[:, 1]

    idx = np.logical_or(data_cost > 0, data_homo > 0)

    data_cost = data_cost[idx]
    data_homo = data_homo[idx]

    data_cost = data_cost - min(data_cost)
    data_cost = data_cost / max(data_cost)

    sorted_idx = np.argsort(data_cost)

    # barplot to show distribution of cuts
    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(10, 10))
    ax1.scatter(data_cost[sorted_idx], data_homo[sorted_idx], alpha=0.2, color=mycolors[0])
    ax1.set_ylim(-0.01, 1.01)
    ax1.set_xlabel('scaled cost of the cut')
    ax1.set_ylabel('homogeneity score of the cut')

    # hist of the cost distribution
    ax2.hist(data_cost, alpha=0.5)
    ax2.set_xlabel('cost of cuts')
    ax2.set_ylabel('number of cuts')

    plt.tight_layout()
    plt.savefig("output/experiments/preprocessing/plot/ideal_quality_of_initial_cuts-" + args['experiment']['dataset_name'] + '_' + 'two' + '.pdf')
    plt.close(fig)

    data, order_function = get_dataset_and_order_function(args)
    cuts = []
    for j in range(1, 71):
        for k in range(70, 100):
            cut = np.zeros(100, dtype=bool)
            cut[0:j] = True
            cut[70:k] = True
            cut_cost = order_function(cut)
            cut_homo = max(metrics.homogeneity_score(data['ys'], cut), metrics.homogeneity_score(cut, data['ys']))
            cuts.append([cut_cost, cut_homo])

    cuts = np.array(cuts)

    data_cost = cuts[:, 0]
    data_homo = cuts[:, 1]

    idx = np.logical_or(data_cost > 0, data_homo > 0)

    data_cost = data_cost[idx]
    data_homo = data_homo[idx]

    data_cost = data_cost - min(data_cost)
    data_cost = data_cost / max(data_cost)

    sorted_idx = np.argsort(data_cost)

    # barplot to show distribution of cuts
    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(10, 10))
    ax1.scatter(data_cost[sorted_idx], data_homo[sorted_idx], alpha=0.2, color=mycolors[0])
    ax1.set_ylim(-0.01, 1.01)
    ax1.set_xlabel('scaled cost of the cut')
    ax1.set_ylabel('homogeneity score of the cut')

    # hist of the cost distribution
    ax2.hist(data_cost, alpha=0.5)
    ax2.set_xlabel('cost of cuts')
    ax2.set_ylabel('number of cuts')

    plt.tight_layout()
    plt.savefig("output/experiments/preprocessing/plot/ideal_quality_of_initial_cuts-" + args['experiment'][
        'dataset_name'] + '_' + 'unbalanced' + '.pdf')
    plt.close(fig)

    data, order_function = get_dataset_and_order_function(args)
    cuts = []
    for j in range(1, 34, 2):
        for k in range(33, 67, 2):
            for l in range(66, 100, 2):
                cut = np.zeros(100, dtype=bool)
                cut[0:j] = True
                cut[33:k] = True
                cut[66:l] = True
                cut_cost = order_function(cut)
                cut_homo = max(metrics.homogeneity_score(data['ys'], cut), metrics.homogeneity_score(cut, data['ys']))
                cuts.append([cut_cost, cut_homo])

    cuts = np.array(cuts)

    data_cost = cuts[:, 0]
    data_homo = cuts[:, 1]

    idx = np.logical_or(data_cost > 0, data_homo > 0)

    data_cost = data_cost[idx]
    data_homo = data_homo[idx]

    data_cost = data_cost - min(data_cost)
    data_cost = data_cost / max(data_cost)

    sorted_idx = np.argsort(data_cost)

    # barplot to show distribution of cuts
    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(10, 10))
    ax1.scatter(data_cost[sorted_idx], data_homo[sorted_idx], alpha=0.2, color=mycolors[0])
    ax1.set_ylim(-0.01, 1.01)
    ax1.set_xlabel('scaled cost of the cut')
    ax1.set_ylabel('homogeneity score of the cut')

    # hist of the cost distribution
    ax2.hist(data_cost, alpha=0.5)
    ax2.set_xlabel('cost of cuts')
    ax2.set_ylabel('number of cuts')

    plt.tight_layout()
    plt.savefig("output/experiments/preprocessing/plot/ideal_quality_of_initial_cuts-" + args['experiment'][
        'dataset_name'] + '_' + 'three' + '.pdf')
    plt.close(fig)


def quality_of_initial_cuts(args, name):
    nb_cuts = args['preprocessing']['nb_cuts']

    if args['experiment']['dataset_name'] == 'knn_gauss_blobs':
        lb_f_len = 1
    else:
        lb_f_len = len(lb_f)

    out = np.zeros([lb_f_len, nb_cuts])

    data, order_function = get_dataset_and_order_function(args)
    for index_bound in range(lb_f_len):
        args['preprocessing']['lb_f'] = lb_f[index_bound]
        cuts = compute_cuts(data, args, verbose=args['verbose'])['values']
        for index_cut, predicted in enumerate(cuts):
            out[index_bound, index_cut] = max(metrics.homogeneity_score(data['ys'], predicted), metrics.homogeneity_score(predicted, data['ys']))

    np.save("output/experiments/preprocessing/quality_of_initial_cuts-" + args['experiment']['dataset_name'] + '_' + name + '.npy', out)


def plot_quality_of_initial_cuts():
    data_two = np.load("output/experiments/preprocessing/quality_of_initial_cuts-" + 'sbm' + '_' + 'two' + '.npy')
    data_unbalanced = np.load("output/experiments/preprocessing/quality_of_initial_cuts-" + 'sbm' + '_' + 'unbalanced' + '.npy')
    data_three = np.load("output/experiments/preprocessing/quality_of_initial_cuts-" + 'sbm' + '_' + 'three' + '.npy')

    data_names = ['two', 'unbalanced', 'three']

    fig, ax = plt.subplots(1, 4, figsize=(40, 10), sharex='all', sharey='all')
    flat_ax = ax.flat
    for bound_idx, bound in enumerate(lb_f[::2]):
        data = np.vstack([data_two[bound_idx*2, :], data_unbalanced[bound_idx*2, :], data_three[bound_idx*2, :]])

        jitter_x = []
        for i, d in enumerate(data):
            jitter_x.append(np.random.normal(i, 0.04, size=len(d)))

        jitter_x = np.array(jitter_x)

        for d, j, i in zip(data, jitter_x, range(len(data))):
            flat_ax[bound_idx].scatter(d, j, color=mycolors[i], alpha=0.1)
            flat_ax[bound_idx].boxplot(d, positions=[i], boxprops=dict(color=mycolors[i]), vert=False)

        flat_ax[bound_idx].set_title('lower bound of {}'.format(np.round(bound, 2)))
        flat_ax[bound_idx].set_xlabel('homogeneity')
        ytickNames = plt.setp(flat_ax[bound_idx], yticklabels=data_names)
        plt.setp(ytickNames)

    plt.tight_layout()
    plt.savefig("output/experiments/preprocessing/plot/quality_of_initial_cuts-" + 'sbm' + '.pdf')
    plt.close(fig)

    data_two = np.load("output/experiments/preprocessing/quality_of_initial_cuts-" + 'knn_gauss_blobs' + '_' + 'two' + '.npy')
    data_unbalanced = np.load("output/experiments/preprocessing/quality_of_initial_cuts-" + 'knn_gauss_blobs' + '_' + 'unbalanced' + '.npy')
    data_three = np.load("output/experiments/preprocessing/quality_of_initial_cuts-" + 'knn_gauss_blobs' + '_' + 'three' + '.npy')

    fig, ax = plt.subplots(figsize=(10, 10))

    data = np.vstack([data_two[0, :], data_unbalanced[0, :], data_three[0, :]])

    jitter_x = []
    for i, d in enumerate(data):
        jitter_x.append(np.random.normal(i, 0.04, size=len(d)))

    jitter_x = np.array(jitter_x)

    for d, j, i in zip(data, jitter_x, range(len(data))):
        ax.scatter(d, j, color=mycolors[i], alpha=0.1)
        ax.boxplot(d, positions=[i], boxprops=dict(color=mycolors[i]), vert=False)

    ax.set_xlabel('homogeneity')
    ax.set_xlim(-0.01, 1.01)
    ytickNames = plt.setp(ax, yticklabels=data_names)
    plt.setp(ytickNames)

    plt.tight_layout()

    plt.savefig("output/experiments/preprocessing/plot/quality_of_initial_cuts-" + 'knn_gauss_blobs' + '.pdf')
    plt.close(fig)


def comparison_of_initial_cuts(args, name):
    out = np.zeros([len(algorithms), args['preprocessing']['nb_cuts'], runs])

    data, _ = get_dataset_and_order_function(args)
    for index_alg, alg in enumerate(algorithms):
        args['experiment']['preprocessing_name'] = alg
        cuts = compute_cuts(data, args, verbose=args['verbose'])['values']
        for index_cut, predicted in enumerate(cuts):
            out[index_alg, index_cut] = max(metrics.homogeneity_score(data['ys'], predicted), metrics.homogeneity_score(predicted, data['ys']))

    np.save("output/experiments/preprocessing/comparison_of_initial_cuts-" + args['experiment']['dataset_name'] + '_' + name + '.npy', out)


def plot_comparison_of_initial_cuts(name):
    data_sbm = np.load('output/experiments/preprocessing/comparison_of_initial_cuts-sbm_' + name + '.npy')
    data_gauss = np.load('output/experiments/preprocessing/comparison_of_initial_cuts-knn_gauss_blobs_' + name + '.npy')

    data_names = ['sbm\nfid-mat', 'sbm\nrand-projection', 'gauss-blobs\nfid-mat', 'gauss-blobs\nrand-projection']

    data = np.concatenate([data_sbm, data_gauss])

    cleaned_data = []
    jitter_x = []
    for i, d in enumerate(data):
        next_cleaned = d[d > 0]
        cleaned_data.append(next_cleaned)
        jitter_x.append(np.random.normal(i, 0.04, size=len(next_cleaned)))

    cleaned_data = np.array(cleaned_data)
    jitter_x = np.array(jitter_x)

    fig, ax1 = plt.subplots(figsize=(6, 9))
    for d, j, i in zip(cleaned_data, jitter_x, range(len(cleaned_data))):
        ax1.scatter(d, j, color=mycolors[i], alpha=0.1)
        ax1.boxplot(d, positions=[i],  boxprops=dict(color=mycolors[i]), vert=False)

    ax1.set_xlabel('homogeneity')
    ax1.set_xlim(-0.01, 1.01)

    ytickNames = plt.setp(ax1, yticklabels=data_names)
    plt.setp(ytickNames)

    plt.tight_layout()
    plt.savefig('output/experiments/preprocessing/plot/comparison_of_initial_cuts-sbm_' + name + '.pdf')
    plt.close(fig)


def choice_of_cost_function(args, name):
    out = np.zeros([len(cost_functions), 2, args['preprocessing']['nb_cuts']])

    data, _ = get_dataset_and_order_function(args)
    cuts = get_exp_cuts(data, args['preprocessing']['nb_cuts'], verbose=args['verbose'])
    cuts = {'values': np.array(cuts), 'names': None, 'equations': None}

    for index_cost_fun, cost_fun in enumerate(cost_functions):
        print("Choosing cost function: {}".format(cost_fun))
        print('{}/{} cost function: {}\n\t'.format(index_cost_fun+1, len(cost_functions), cost_fun))
        order_function = resolve_cost_function(cost_fun, data)
        all_cuts, all_orders = order_cuts(cuts, order_function)

        homogeneity = []
        for c in all_cuts['values']:
            homogeneity.append(max(metrics.homogeneity_score(data['ys'], c), metrics.homogeneity_score(c, data['ys'])))

        out[index_cost_fun, 0, :] = homogeneity
        out[index_cost_fun, 1, :] = all_orders

    np.save("output/experiments/tangles/cost_function-" + args['experiment']['dataset_name'] + '_' + name + '.npy', out)


def plot_choice_of_cost_function(dataset):

    if dataset == 'sbm':
        cost_funs = [1, 3]
    else:
        cost_funs = [0, 2]
    data_sbm_two = np.load("output/experiments/tangles/cost_function-" + dataset + '_' + 'two' + '.npy')
    data_sbm_unbalanced = np.load("output/experiments/tangles/cost_function-" + dataset + '_' + 'unbalanced' + '.npy')
    data_sbm_three = np.load("output/experiments/tangles/cost_function-" + dataset + '_' + 'three' + '.npy')

    fig, ax = plt.subplots(1, 2, figsize=(20, 10), sharey='all')

    for idx_axis, idx_cost_fun in enumerate(cost_funs):
        cost_data_three = data_sbm_three[idx_cost_fun, 1]
        cost_data_three = cost_data_three - min(cost_data_three)
        cost_data_three = cost_data_three / max(cost_data_three)
        ax[idx_axis].scatter(cost_data_three, data_sbm_three[idx_cost_fun, 0], color=mycolors[1], marker=mymarkers[1], alpha=0.4, label='three')
        cost_data_two = data_sbm_two[idx_cost_fun, 1]
        cost_data_two = cost_data_two - min(cost_data_two)
        cost_data_two = cost_data_two / max(cost_data_two)
        ax[idx_axis].scatter(cost_data_two, data_sbm_two[idx_cost_fun, 0], color=mycolors[0], marker=mymarkers[0], alpha=0.4, label='two')
        cost_data_unbalanced = data_sbm_unbalanced[idx_cost_fun, 1]
        cost_data_unbalanced = cost_data_unbalanced - min(cost_data_unbalanced)
        cost_data_unbalanced = cost_data_unbalanced / max(cost_data_unbalanced)
        ax[idx_axis].scatter(cost_data_unbalanced, data_sbm_two[idx_cost_fun, 0], color=mycolors[2], marker=mymarkers[2],
                             alpha=0.4, label='unbalanced')
        ax[idx_axis].set_xlabel('normalized cost')
        ax[idx_axis].set_title(cost_functions[idx_cost_fun])

    ax[0].set_ylim(-0.01, 1.01)
    ax[1].legend()
    ax[0].set_ylabel('homogeneity')

    plt.tight_layout()
    plt.savefig('output/experiments/tangles/plot/cost_fun-' + dataset + '.pdf')
    plt.close(fig)


def choice_of_a(args, name):
    out = np.zeros([len(agreement), runs])

    pool = multiprocessing.Pool()
    for r in range(runs):
        args['experiment']['seed'] = (r+1) * 10
        data, order_function = get_dataset_and_order_function(args)
        cuts = get_exp_cuts(data, args['preprocessing']['nb_cuts'], verbose=args['verbose'])
        cuts = {'values': np.array(cuts), 'names': None, 'equations': None}
        all_cuts, all_orders = order_cuts(cuts, order_function)

        vms, _ = zip(*pool.map(partial(run_for_a, all_cuts, all_orders, data), agreement))

        out[:, r] = vms

    pool.close()
    np.save("output/experiments/tangles/a-" + args['experiment']['dataset_name'] + '_' + name + '.npy', out)


def run_for_a(all_cuts, all_orders, data, a):
    model = TangleTreeModel(agreement=a, cuts=all_cuts["values"], costs=all_orders,
                            weight_fun=sigmoid)

    # evaluate tangle output
    leaves = np.array(model.maximals)
    if len(leaves) > 0:
        probs_leaves = np.stack(np.array(leaves[:, 0]))
        tangle_labels = np.argmax(probs_leaves, axis=0)

        vms_tangles = v_measure_score(data["ys"], tangle_labels)
        homo_tangle = metrics.homogeneity_score(data["ys"], tangle_labels)
    else:
        homo_tangle = 0
        vms_tangles = 0

    return vms_tangles, homo_tangle


def plot_choice_of_a():
    data_gauss_two = np.load("output/experiments/tangles/a-" + 'knn_gauss_blobs' + '_' + 'two' + '.npy')
    data_gauss_unbalanced = np.load("output/experiments/tangles/a-" + 'knn_gauss_blobs' + '_' + 'unbalanced' + '.npy')
    data_gauss_three = np.load("output/experiments/tangles/a-" + 'knn_gauss_blobs' + '_' + 'three' + '.npy')

    data_sbm_two = np.load("output/experiments/tangles/a-" + 'sbm' + '_' + 'two' + '.npy')
    data_sbm_unbalanced = np.load("output/experiments/tangles/a-" + 'sbm' + '_' + 'unbalanced' + '.npy')
    data_sbm_three = np.load("output/experiments/tangles/a-" + 'sbm' + '_' + 'three' + '.npy')

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(20, 10))

    mean = data_gauss_two.mean(axis=1)
    std = data_gauss_two.std(axis=1)
    ax1.plot(agreement, mean, color=mycolors[0], label='two', marker=mymarkers[0])
    ax1.fill_between(agreement, mean+std, mean-std, facecolor=mycolors[0], alpha=0.3)
    mean = data_gauss_unbalanced.mean(axis=1)
    std = data_gauss_unbalanced.std(axis=1)
    ax1.plot(agreement, mean, color=mycolors[1], label='unbalanced', marker=mymarkers[1])
    ax1.fill_between(agreement, mean + std, mean - std, facecolor=mycolors[1], alpha=0.3)
    mean = data_gauss_three.mean(axis=1)
    std = data_gauss_three.std(axis=1)
    ax1.plot(agreement, mean, color=mycolors[2], label='three', marker=mymarkers[2])
    ax1.fill_between(agreement, mean + std, mean - std, facecolor=mycolors[2], alpha=0.3)

    ax1.set_xlabel('agreemet')
    ax1.set_ylabel('v measure score')

    ax1.set_title('gaussian mixture model')
    ax1.set_ylim(-0.01, 1.01)
    ax1.legend()

    mean = data_sbm_two.mean(axis=1)
    std = data_sbm_two.std(axis=1)
    ax2.plot(agreement, mean, marker=mymarkers[0], color=mycolors[0], label='two')
    ax2.fill_between(agreement, mean+std, mean-std, facecolor=mycolors[0], alpha=0.3)
    mean = data_sbm_unbalanced.mean(axis=1)
    std = data_sbm_unbalanced.std(axis=1)
    ax2.plot(agreement, mean, marker=mymarkers[1], color=mycolors[1], label='unbalanced')
    ax2.fill_between(agreement, mean + std, mean - std, facecolor=mycolors[1], alpha=0.3)
    mean = data_sbm_three.mean(axis=1)
    std = data_sbm_three.std(axis=1)
    ax2.plot(agreement, mean, marker=mymarkers[2], color=mycolors[2], label='three')
    ax2.fill_between(agreement, mean + std, mean - std, facecolor=mycolors[2], alpha=0.3)

    ax2.set_ylim(-0.01, 1.01)
    ax2.set_xlabel('agreemet')
    ax2.set_ylabel('v measure score')

    ax2.set_title('stochastic block model')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('output/experiments/tangles/plot/a.pdf')
    plt.close(fig)


def number_of_cuts(args, name):
    out = np.zeros([len(nb_cuts), runs])
    args['preprocessing']['nb_cuts'] = max(nb_cuts)

    pool = multiprocessing.Pool()
    for r in range(runs):
        args['experiment']['seed'] = (r + 1) * 10
        data, order_function = get_dataset_and_order_function(args)
        cuts = compute_cuts(data, args, verbose=args['verbose'])

        out[:, r] = pool.map(partial(run_for_nb_cuts, cuts, order_function, args, data), nb_cuts)

    pool.close()

    np.save("output/experiments/tangles/nb_cuts-" + args['experiment']['dataset_name'] + '_' + name + '.npy', out)


def run_for_nb_cuts(cuts, order_function, args, data, nb_cuts):
    cuts = {'values': cuts['values'][:nb_cuts], 'names': None, 'equations': None}
    cuts, orders = order_cuts(cuts, order_function)
    all_cuts, all_orders = pick_cuts_up_to_order(deepcopy(cuts), deepcopy(orders),
                                                 percentile=args['experiment']['percentile_orders'])
    model = TangleTreeModel(agreement=args["experiment"]["agreement"], cuts=all_cuts["values"], costs=all_orders,
                            weight_fun=sigmoid)

    # evaluate tangle output
    leaves = np.array(model.maximals)
    if len(leaves) > 0:
        probs_leaves = np.stack(np.array(leaves[:, 0]))
        tangle_labels = np.argmax(probs_leaves, axis=0)

        vms_tangles = v_measure_score(data["ys"], tangle_labels)
    else:
        vms_tangles = 0

    return  vms_tangles


def plot_number_of_cuts():

    data_gauss_two = np.load("output/experiments/tangles/nb_cuts-" + 'knn_gauss_blobs' + '_' + 'two' + '.npy')
    data_gauss_unbalanced = np.load("output/experiments/tangles/nb_cuts-" + 'knn_gauss_blobs' + '_' + 'unbalanced' + '.npy')
    data_gauss_three = np.load("output/experiments/tangles/nb_cuts-" + 'knn_gauss_blobs' + '_' + 'three' + '.npy')

    data_sbm_two = np.load("output/experiments/tangles/nb_cuts-" + 'sbm' + '_' + 'two' + '.npy')
    data_sbm_unbalanced = np.load("output/experiments/tangles/nb_cuts-" + 'sbm' + '_' + 'unbalanced' + '.npy')
    data_sbm_three = np.load("output/experiments/tangles/nb_cuts-" + 'sbm' + '_' + 'three' + '.npy')

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(20, 10))

    mean = data_gauss_two.mean(axis=1)
    std = data_gauss_two.std(axis=1)
    ax1.plot(nb_cuts, mean, marker=mymarkers[0], color=mycolors[0], label='two')
    ax1.fill_between(nb_cuts, mean + std, mean - std, facecolor=mycolors[0], alpha=0.3)
    mean = data_gauss_unbalanced.mean(axis=1)
    std = data_gauss_unbalanced.std(axis=1)
    ax1.plot(nb_cuts, mean, marker=mymarkers[1], color=mycolors[1], label='unbalanced')
    ax1.fill_between(nb_cuts, mean + std, mean - std, facecolor=mycolors[1], alpha=0.3)
    mean = data_gauss_three.mean(axis=1)
    std = data_gauss_three.std(axis=1)
    ax1.plot(nb_cuts, mean, marker=mymarkers[2], color=mycolors[2], label='three')
    ax1.fill_between(nb_cuts, mean + std, mean - std, facecolor=mycolors[2], alpha=0.3)

    ax1.set_xlabel('nb of cuts')
    ax1.set_ylabel('v measure score')

    ax1.set_title('gaussian mixture model')
    ax1.set_ylim(-0.01, 1.01)
    ax1.legend()

    mean = data_sbm_two.mean(axis=1)
    std = data_sbm_two.std(axis=1)
    ax2.plot(nb_cuts, mean, marker=mymarkers[0], color=mycolors[0], label='two')
    ax2.fill_between(nb_cuts, mean+std, mean-std, facecolor=mycolors[0], alpha=0.3)
    mean = data_sbm_unbalanced.mean(axis=1)
    std = data_sbm_unbalanced.std(axis=1)
    ax2.plot(nb_cuts, mean, marker=mymarkers[1], color=mycolors[1], label='unbalanced')
    ax2.fill_between(nb_cuts, mean + std, mean - std, facecolor=mycolors[1], alpha=0.3)
    mean = data_sbm_three.mean(axis=1)
    std = data_sbm_three.std(axis=1)
    ax2.plot(nb_cuts, mean, marker=mymarkers[2], color=mycolors[2], label='three')
    ax2.fill_between(nb_cuts, mean + std, mean - std, facecolor=mycolors[2], alpha=0.3)

    ax2.set_xlabel('nb of cuts')
    ax2.set_ylabel('v measure score')
    ax2.set_ylim(-0.01, 1.01)


    ax2.set_title('stochastic block model')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('output/experiments/tangles/plot/nb_cuts.pdf')
    plt.close(fig)


def influence_of_cut_quality(args, name):
    out = np.zeros([len(quality), 4, runs])

    pool = multiprocessing.Pool()

    for r in range(runs):
        print('run {}/{}'.format(r + 1, runs))
        seed = (r + 1) * 10
        np.random.seed(seed)
        args['experiment']['seed'] = seed
        data, order_function = get_dataset_and_order_function(args)

        vms_tangles, homo_tangles, homogeneity_mean, homogeneity_std = zip(*pool.map(partial(run_for_quali, data, order_function, args), quality))

        out[:, 0, r] = vms_tangles
        out[:, 1, r] = homo_tangles
        out[:, 2, r] = homogeneity_mean
        out[:, 3, r] = homogeneity_std

    pool.close()
    np.save("output/experiments/tangles/cut_quality-" + args['experiment']['dataset_name'] + '-' + name + '.npy', out)


def run_for_quali(data, order_function, args, quali):
    nb_nodes = len(data['ys'])
    classes = np.unique(data['ys'])
    nb_classes = len(classes)
    cut_values = []
    homogeneity = []
    for c in range(args['preprocessing']['nb_cuts']):
        cut_tmp = get_cuts_of_quality(data, quality=np.random.choice(quali), seed=(c+1) * 10)
        cut_values.append(cut_tmp)
        homogeneity.append(max(metrics.homogeneity_score(data['ys'], cut_tmp),
                              metrics.homogeneity_score(cut_tmp, data['ys'])))

    cuts = {'values': np.array(cut_values), 'names': None, 'equations': None}
    cuts, orders = order_cuts(cuts, order_function)
    all_cuts, all_orders = pick_cuts_up_to_order(deepcopy(cuts), deepcopy(orders),
                                                 percentile=args['experiment']['percentile_orders'])
    model = TangleTreeModel(agreement=args["experiment"]["agreement"], cuts=all_cuts["values"], costs=all_orders,
                            weight_fun=sigmoid)

    # evaluate tangle output
    leaves = np.array(model.maximals)
    if len(leaves) > 0:
        probs_leaves = np.stack(np.array(leaves[:, 0]))
        tangle_labels = np.argmax(probs_leaves, axis=0)

        vms_tangles = v_measure_score(data["ys"], tangle_labels)
        homo_tangle = metrics.homogeneity_score(data['ys'], tangle_labels)
    else:
        homo_tangle = 0
        vms_tangles = 0

    scatter_homo = np.array([max(metrics.homogeneity_score(data['ys'], c),
                           metrics.homogeneity_score(c, data['ys'])) for c in all_cuts['values']])

    plt.figure()
    plt.scatter(all_orders, scatter_homo)
    plt.title('HOMOGENEITY - mean cuts: {}, tangle: {}'.format(np.array(homogeneity).mean(), vms_tangles))
    plt.savefig('quali_{}.pdf'.format(quali))
    plt.close()

    return vms_tangles, homo_tangle, np.array(homogeneity).mean(), np.array(homogeneity).std()


def plot_influence_of_cut_quality(dataset, name):
    data = np.load("output/experiments/tangles/cut_quality-" + dataset + '-' + name + '.npy')

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.errorbar(data[0, 2, :].mean(), data[0, 0, :].mean(), xerr=data[0, 2, :].std(), yerr=data[0, 0, :].std(),
                label='flipped {} of the nodes'.format(quality[0]), color=mycolors[0], marker=mymarkers[0])
    #ax.errorbar(data[1, 2, :].mean(), data[1, 0, :].mean(), xerr=data[1, 2, :].std(), yerr=data[1, 0, :].std(),
    #            label='flipped {} of the nodes'.format(quality[1]), color=mycolors[1], marker=mymarkers[1])
    ax.errorbar(data[2, 2, :].mean(), data[2, 0, :].mean(), xerr=data[2, 2, :].std(), yerr=data[2, 0, :].std(),
                label='flipped {} of the nodes'.format(quality[2]), color=mycolors[2], marker=mymarkers[2])
    #ax.errorbar(data[3, 2, :].mean(), data[3, 0, :].mean(), xerr=data[3, 2, :].std(), yerr=data[3, 0, :].std(),
    #            label='flipped {} of the nodes'.format(quality[3]), color=mycolors[3], marker=mymarkers[3])
    ax.errorbar(data[4, 2, :].mean(), data[3, 0, :].mean(), xerr=data[4, 2, :].std(), yerr=data[4, 0, :].std(),
                label='flipped {} of the nodes'.format(quality[4]), color=mycolors[4], marker=mymarkers[4])

    ax.set_xlabel('mean homogeneity of cuts')
    ax.set_ylabel('homogeneity of clustering')
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlim(-0.01, 1.01)
    ax.legend()

    plt.tight_layout()
    plt.savefig('output/experiments/tangles/plot/influence_cut_quality-' + dataset + '-' + name + '.pdf')
    plt.close(fig)


def choice_of_psi(args, name):
    out = np.zeros([len(psis), runs])

    pool = multiprocessing.Pool()
    for r in range(runs):
        args['experiment']['seed'] = (r+1) * 10
        data, order_function = get_dataset_and_order_function(args)
        cuts = get_exp_cuts(data, args['preprocessing']['nb_cuts'], verbose=args['verbose'])
        cuts = {'values': np.array(cuts), 'names': None, 'equations': None}
        all_cuts, all_orders = order_cuts(cuts, order_function)

        out[:, r], _ = zip(*pool.map(partial(run_for_psi, all_cuts, all_orders, args, data), psis))

    pool.close()
    np.save("output/experiments/postprocessing/psi-" + args['experiment']['dataset_name'] + '_' + name + '.npy', out)


def run_for_psi(cuts, orders, args, data, psi):
    all_cuts, all_orders = pick_cuts_up_to_order(cuts, orders, percentile=psi)
    if len(all_cuts['values']) == 0:
        vms_tangles = 0
        homo_tangle = 0
    else:
        model = TangleTreeModel(agreement=args["experiment"]["agreement"], cuts=all_cuts["values"], costs=all_orders,
                                weight_fun=sigmoid)
        # evaluate tangle output
        leaves = np.array(model.maximals)
        if len(leaves) > 0:
            probs_leaves = np.stack(np.array(leaves[:, 0]))
            tangle_labels = np.argmax(probs_leaves, axis=0)

            vms_tangles = v_measure_score(data["ys"], tangle_labels)
            homo_tangle = metrics.homogeneity_score(data["ys"], tangle_labels)
        else:
            homo_tangle = 0
            vms_tangles = 0

    return vms_tangles, homo_tangle


def plot_choice_of_psi():
    data_gauss_two = np.load("output/experiments/postprocessing/psi-" + 'knn_gauss_blobs' + '_' + 'two' + '.npy')
    data_gauss_unbalanced = np.load("output/experiments/postprocessing/psi-" + 'knn_gauss_blobs' + '_' + 'unbalanced' + '.npy')
    data_gauss_three = np.load("output/experiments/postprocessing/psi-" + 'knn_gauss_blobs' + '_' + 'three' + '.npy')

    data_sbm_two = np.load("output/experiments/postprocessing/psi-" + 'sbm' + '_' + 'two' + '.npy')
    data_sbm_unbalanced = np.load("output/experiments/postprocessing/psi-" + 'sbm' + '_' + 'unbalanced' + '.npy')
    data_sbm_three = np.load("output/experiments/postprocessing/psi-" + 'sbm' + '_' + 'three' + '.npy')

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(20, 10))

    mean = data_gauss_two.mean(axis=1)
    std = data_gauss_two.std(axis=1)
    ax1.plot(psis, mean, marker=mymarkers[0], color=mycolors[0], label='two')
    ax1.fill_between(psis, mean + std, mean - std, facecolor=mycolors[0], alpha=0.3)
    mean = data_gauss_unbalanced.mean(axis=1)
    std = data_gauss_unbalanced.std(axis=1)
    ax1.plot(psis, mean, marker=mymarkers[1], color=mycolors[1], label='unbalanced')
    ax1.fill_between(psis, mean + std, mean - std, facecolor=mycolors[1], alpha=0.3)
    mean = data_gauss_three.mean(axis=1)
    std = data_gauss_three.std(axis=1)
    ax1.plot(psis, mean, marker=mymarkers[2], color=mycolors[2], label='three')
    ax1.fill_between(psis, mean + std, mean - std, facecolor=mycolors[2], alpha=0.3)
    ax1.set_ylim(-0.01, 1.01)

    ax1.set_xlabel('percentile of considered orders')
    ax1.set_ylabel('v measure score')

    ax1.set_title('gaussian mixture model')
    ax1.legend()

    mean = data_sbm_two.mean(axis=1)
    std = data_sbm_two.std(axis=1)
    ax2.plot(psis, mean, marker=mymarkers[0], color=mycolors[0], label='two')
    ax2.fill_between(psis, mean+std, mean-std, facecolor=mycolors[0], alpha=0.3)
    mean = data_sbm_unbalanced.mean(axis=1)
    std = data_sbm_unbalanced.std(axis=1)
    ax2.plot(psis, mean, marker=mymarkers[1], color=mycolors[1], label='unbalanced')
    ax2.fill_between(psis, mean + std, mean - std, facecolor=mycolors[1], alpha=0.3)
    mean = data_sbm_three.mean(axis=1)
    std = data_sbm_three.std(axis=1)
    ax2.plot(psis, mean, marker=mymarkers[2], color=mycolors[2], label='three')
    ax2.fill_between(psis, mean + std, mean - std, facecolor=mycolors[2], alpha=0.3)
    ax2.set_ylim(-0.01, 1.01)

    ax2.set_xlabel('percentile of considered orders')
    ax2.set_ylabel('v measure score')

    ax2.set_title('stochastic block model')

    ax2.legend()
    plt.tight_layout()
    plt.savefig('output/experiments/postprocessing/plot/psi.pdf')
    plt.close(fig)


def interplay_cut_quality_and_a(args, name):
    out = np.zeros([len(agreement), len(quality), 4, runs])

    pool = multiprocessing.Pool()
    for index_a, a in enumerate(agreement):
        args['experiment']['agreement'] = a
        for r in range(runs):
            seed = (r + 1) * 10
            print('{}/{} agreement: {}\n\t run {}/{}'.format(index_a + 1, len(agreement), a, r + 1,
                                                                 runs))
            args['experiment']['seed'] = seed
            data, order_function = get_dataset_and_order_function(args)

            out[index_a, :, 0, r], out[index_a, :, 1, r], out[0, :, 2, r], out[0, :, 3, r] = zip(*pool.map(partial(run_for_quali, data, order_function, args), quality))

    np.save("output/experiments/interplay/cut_quality_and_a-" + args['experiment']['dataset_name'] + '-' + name + '.npy', out)


def plot_interplay_cut_quality_and_a(dataset, name):
    data = np.load("output/experiments/interplay/cut_quality_and_a-" + dataset + '-' + name + '.npy')

    fig, ax = plt.subplots(figsize=(10, 10))

    for q_index, q in enumerate(quality[::2]):

        mean = data[:, q_index*2, 0, :].mean(axis=1)
        std = data[:, q_index*2, 0, :].std(axis=1)
        ax.plot(agreement, mean, color=mycolors[q_index], label='flipped {} the points'.format(q), marker=mymarkers[q_index])
        ax.fill_between(agreement, mean+std, mean-std, facecolor=mycolors[q_index], alpha=0.3)

    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel('agreement')
    ax.set_ylabel('v measure score')

    ax.legend()
    plt.tight_layout()
    plt.savefig("output/experiments/interplay/plot/cut_quality_and_a-" + dataset + '-' + name + '.pdf')
    plt.close(fig)


def interplay_cut_quality_and_psi(args, name):
    out = np.zeros([len(psis), len(quality), 4, runs])

    pool = multiprocessing.Pool()
    for index_psi, psi in enumerate(psis):
        args['experiment']['percentile_orders'] = psi
        for r in range(runs):
            print('{}/{} psi: {}\n\t run {}/{}'.format(index_psi + 1, len(psis), psi, r + 1,
                                                                 runs))
            seed = (r + 1) * 10
            args['experiment']['seed'] = seed
            data, order_function = get_dataset_and_order_function(args)

            out[index_psi, :, 0, r], out[index_psi, :, 1, r], out[index_psi, :, 2, r], out[index_psi, :, 3, r] = zip(*pool.map(partial(run_for_quali, data, order_function, args), quality))


    np.save("output/experiments/interplay/cut_quality_and_psi-" + args['experiment']['dataset_name'] + '-' +  name + '.npy', out)


def plot_interplay_cut_quality_and_psi(dataset, name):
    data = np.load("output/experiments/interplay/cut_quality_and_psi-" + dataset + '-' + name + '.npy')

    fig, ax = plt.subplots(figsize=(10, 10))
    for q_index, q in enumerate(quality[::2]):

        mean = data[:, q_index*2, 0, :].mean(axis=1)
        std = data[:, q_index*2, 0, :].std(axis=1)
        ax.plot(psis, mean, color=mycolors[q_index], label='flipped {} the points'.format(q), marker=mymarkers[q_index])
        ax.fill_between(psis, mean + std, mean - std, facecolor=mycolors[q_index], alpha=0.3)

    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel('psi')
    ax.set_ylabel('v measure score')

    ax.legend()
    plt.tight_layout()
    plt.savefig("output/experiments/interplay/plot/cut_quality_and_psi-" + dataset + '-' + name + '.pdf')
    plt.close(fig)


def interplay_a_and_psi(args, name):
    out = np.zeros([len(agreement), len(psis), runs])

    pool = multiprocessing.Pool()
    for r in range(runs):
        args['experiment']['seed'] = (r + 1) * 10
        data, order_function = get_dataset_and_order_function(args)
        cuts = get_exp_cuts(data, args['preprocessing']['nb_cuts'], verbose=args['verbose'])
        cuts = {'values': np.array(cuts), 'names': None, 'equations': None}
        cuts, orders = order_cuts(cuts, order_function)
        for index_psi, psi in enumerate(psis):
            all_cuts, all_orders = pick_cuts_up_to_order(deepcopy(cuts), deepcopy(orders), percentile=psi)
            if len(all_cuts['values']) == 0:
                out[:, index_psi, r] = 0
            else:
                out[:, index_psi, r], _ = zip(*pool.map(partial(run_for_a, all_cuts, all_orders, data), agreement))

    np.save("output/experiments/interplay/a_and_psi-" + args['experiment']['dataset_name'] + '_' + name + '.npy', out)


def plot_interplay_a_and_psi(dataset, name):
    data = np.load("output/experiments/interplay/a_and_psi-" + dataset + '_' + name + '.npy')

    fig, ax = plt.subplots(figsize=(10, 10))
    for a_index, a in enumerate(agreement[::3]):

        mean = data[a_index*3, :, :].mean(axis=1)
        std = data[a_index*3, :, :].std(axis=1)
        ax.plot(psis, mean, color=mycolors[a_index], label='agreement: {}'.format(agreement[a_index*2]), marker=mymarkers[a_index])
        ax.fill_between(psis, mean+std, mean-std, facecolor=mycolors[a_index], alpha=0.3)

    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel('psi')
    ax.set_ylabel('v measure score')

    ax.legend()
    plt.tight_layout()
    plt.savefig("output/experiments/interplay/plot/a_and_psi-" + dataset + '_' + name + '.pdf')
    plt.close(fig)


def psi_tree(args, name):
    pool = multiprocessing.Pool()

    out, _ = zip(*pool.map(partial(evaluate_all, args), list(range(runs))))

    pool.close()
    np.save("output/experiments/postprocessing/psi_tree-" + args['experiment']['dataset_name'] + '_' + name + '.npy', out)


def evaluate_all(args, r):

    args['experiment']['seed'] = (r + 1) * 10
    data, order_function = get_dataset_and_order_function(args)
    cuts = get_exp_cuts(data, args['preprocessing']['nb_cuts'], verbose=args['verbose'])
    cuts = {'values': np.array(cuts), 'names': None, 'equations': None}
    all_cuts, all_orders = order_cuts(cuts, order_function)
    model = TangleTreeModel(agreement=args['experiment']['agreement'], cuts=all_cuts["values"], costs=all_orders,
                            weight_fun=sigmoid)

    vms_tree = 0
    homo_tree = 0
    for p in np.arange(5, 101, 5):
        cuts, orders = pick_cuts_up_to_order(deepcopy(all_cuts), deepcopy(all_orders), p)
        if len(cuts['values']) == 0:
            continue
        model = TangleTreeModel(agreement=args['experiment']['agreement'], cuts=cuts["values"], costs=orders,
                                weight_fun=sigmoid)

        leaves = np.array(model.maximals)

        if len(leaves) > 0:
            probs_leaves = np.stack(np.array(leaves[:, 0]))
            tangle_labels = np.argmax(probs_leaves, axis=0)

            vms_tree = max(vms_tree, v_measure_score(data["ys"], tangle_labels))
            homo_tree = max(homo_tree, metrics.homogeneity_score(data["ys"], tangle_labels))

    return vms_tree, homo_tree


def plot_psi_tree():
    data = []
    names = []
    data.append(np.load("output/experiments/postprocessing/psi_tree-" + 'knn_gauss_blobs' + '_' + 'two' + '.npy'))
    names.append('gauss - two')
    data.append(np.load("output/experiments/postprocessing/psi_tree-" + 'knn_gauss_blobs' + '_' + 'unbalanced' + '.npy'))
    names.append('gauss - unbalanced')
    data.append(np.load("output/experiments/postprocessing/psi_tree-" + 'knn_gauss_blobs' + '_' + 'three' + '.npy'))
    names.append('gauss - three')

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(20, 10), sharey='all')

    jitter_x = []
    for i, d in enumerate(data):
        jitter_x.append(np.random.normal(i, 0.04, size=len(d)))

    data = np.array(data)
    jitter_x = np.array(jitter_x)

    for d, j, i in zip(data, jitter_x, range(len(data))):
        ax1.scatter(j, d, color=mycolors[i], alpha=0.4, label=names[i])
        ax1.boxplot(d, positions=[i],  boxprops=dict(color=mycolors[i]))

    ax1.set_ylabel('v measure score')
    ax1.set_ylim(0, 1.05)

    ax1.set_title('gaussian mixture model')
    ax1.legend()

    data = []
    names = []
    data.append(np.load("output/experiments/postprocessing/psi_tree-" + 'sbm' + '_' + 'two' + '.npy'))
    names.append('sbm - two')
    data.append(np.load("output/experiments/postprocessing/psi_tree-" + 'sbm' + '_' + 'unbalanced' + '.npy'))
    names.append('sbm - unbalanced')
    data.append(np.load("output/experiments/postprocessing/psi_tree-" + 'sbm' + '_' + 'three' + '.npy'))
    names.append('sbm - three')

    jitter_x = []
    for i, d in enumerate(data):
        jitter_x.append(np.random.normal(i, 0.04, size=len(d)))

    data = np.array(data)
    jitter_x = np.array(jitter_x)

    for d, j, i in zip(data, jitter_x, range(len(data))):
        ax2.scatter(j, d, color=mycolors[i], alpha=0.4, label=names[i])
        ax2.boxplot(d, positions=[i], boxprops=dict(color=mycolors[i]))

    ax2.set_title('stochastic block model')
    ax2.legend()

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)

    plt.tight_layout()
    plt.savefig('output/experiments/postprocessing/plot/psi_tree.pdf')
    plt.close(fig)


def get_uniform_cuts(data, nb_cuts, verbose):
    cut_values = []
    if verbose > 2:
        print('calculating {} cuts'.format(nb_cuts))
    for c in range(nb_cuts):
        if verbose > 2:
            print('cut {}/{}'.format(c+1, nb_cuts))
        lamb = 5
        q = -np.log(1 - np.exp(-lamb) * np.random.uniform(0, 0.5)) / lamb
        cut_values.append(get_cuts_of_quality(data, q))

    return cut_values


def get_exp_cuts(data, nb_cuts, verbose):
    cut_values = []
    if verbose > 2:
        print('calculating {} cuts'.format(nb_cuts))
    for c in range(nb_cuts):
        if verbose > 2:
            print('cut {}/{}'.format(c+1, nb_cuts))

        my_exp = lambda x:  (np.log(x + 0.1) + 2.25) / 4.75
        q = my_exp(np.random.uniform(0, 1))
        cut_values.append(get_cuts_of_quality(data, q))

    return cut_values


def get_cuts_of_quality(data, quality, seed):
    nb_nodes = len(data['ys'])
    classes = np.unique(data['ys'])
    nb_classes = len(classes)

    np.random.seed(seed)

    np.random.shuffle(classes)
    idx = classes[:np.random.choice(np.arange(1, nb_classes))]
    cut_tmp = np.array([True if point in idx else False for point in data['ys']])
    size_one = min(sum(cut_tmp), sum(~cut_tmp))
    arg_one = np.argmin([sum(cut_tmp), sum(~cut_tmp)])
    if int(quality * size_one) == 0:
        return cut_tmp
    k_size = np.random.randint(0, min(size_one, int(quality*nb_nodes)))
    j_size = int(nb_nodes * quality) - k_size
    if arg_one:
        k_idx = np.random.choice(np.arange(nb_nodes)[~cut_tmp], k_size, replace=False)
        j_idx = np.random.choice(np.arange(nb_nodes)[cut_tmp], j_size, replace=False)
    else:
        k_idx = np.random.choice(np.arange(nb_nodes)[cut_tmp], k_size, replace=False)
        j_idx = np.random.choice(np.arange(nb_nodes)[~cut_tmp], j_size, replace=False)
    cut_tmp[k_idx] = ~cut_tmp[k_idx].astype(bool)
    cut_tmp[j_idx] = ~cut_tmp[j_idx].astype(bool)

    return cut_tmp
