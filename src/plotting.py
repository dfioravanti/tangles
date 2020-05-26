import numpy as np
import networkx as nx

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.preprocessing import minmax_scale

from src.config import NAN
from src.utils import get_points_to_plot, normalize


# Standard colors for uniform plots
COLOR_SILVER = '#C0C0C0'
COLOR_SILVER_RGB = (192 / 255, 192 / 255, 192 / 255) + (0.2,)
COLOR_INDIGO_RGB = (55 / 255, 0 / 255, 175 / 255) + (0.5,)
COLOR_CARNATION_RGB = np.array((247 / 255, 96 / 255, 114 / 255, 1)).reshape((1, -1))
CMAP = plt.cm.get_cmap('Blues')

# TODO: Fix the comments in this file

def get_nb_points(data):

    if data['xs'] is not None:
        return len(data['xs'])
    elif data['A'] is not None:
        return len(data['A'])
    else:
        raise KeyError('What data are you using?')

def append_to_binary(number, new_digit):

    return int(str(bin(number) + str(new_digit)), 2)

def get_next_id(current_id, direction):

    if current_id == 0:
        if direction == 'left':
            return 1
        else:
            return 2
            
    level = int(np.ceil(np.log2(current_id)))

    if direction == 'left':
        return current_id + 2 ** level + 1
    else:
        return current_id + 2 ** level + 2

def plot_dataset(data, colors, ax=None, cmap=None, add_colorbar=True, pos=None):

    if data['xs'] is not None:
        ax = plot_dataset_metric(data['xs'], data['cs'], colors, ax, cmap, add_colorbar)
    elif data['G'] is not None:
        ax, pos = plot_dataset_graph(data['G'], data['ys'], colors, ax, cmap, add_colorbar, pos)

    return ax, pos


def add_colorbar_to_ax(ax, cmap):

    cb = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=1), cmap=cmap),
                          ax=ax, orientation='vertical')   
    cb.ax.set_title('p', y=-.05)

    return ax


def plot_dataset_graph(G, ys, colors, ax, cmap, add_colorbar, pos):

    if pos is None:
        pos = get_position(G, ys)

    nx.draw_networkx(G, pos=pos, ax=ax, node_color=colors, edge_color=COLOR_SILVER, with_labels=False, edgecolors='black')
    if add_colorbar:
        ax = add_colorbar_to_ax(ax, cmap)

    return ax, pos


def plot_dataset_metric(xs, cs, colors, ax, cmap, add_colorbar):

    plt.style.use('ggplot')
    plt.ioff()

    ax.tick_params(axis='x', colors=(0,0,0,0))
    ax.tick_params(axis='y', colors=(0,0,0,0))
    ax.grid(True)

    xs_embedded, cs_embedded = get_points_to_plot(xs, cs)

    sc = ax.scatter(xs_embedded[:, 0], xs_embedded[:, 1], color=colors, vmin=0, vmax=1, edgecolor='black')
    if add_colorbar:
        ax = add_colorbar_to_ax(ax, cmap)

    return ax

def labels_to_colors(ys, cmap):
    
    nb_points = len(ys)
    colors = np.zeros((nb_points, 4))
    normalize_ys = mpl.colors.Normalize(vmin=0, vmax=np.max(ys))

    for y in np.unique(ys):
        idx_current = (ys == y).nonzero()[0]
        color = cmap(normalize_ys(y))
        colors[idx_current, :] = np.array(color).reshape((1, -1))

    return colors        


def plot_soft_predictions(data, contracted_tree, id_node=0, path=None):

    plt.style.use('ggplot')
    plt.ioff()

    cmap_groundtruth = plt.cm.get_cmap('tab10')
    cmap_heatmap = plt.cm.get_cmap('Blues')

    if path is not None:
        output_path = path
        output_path.mkdir(parents=True, exist_ok=True)

    if data['ys'] is not None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
        colors = labels_to_colors(data['ys'], cmap=cmap_groundtruth)
        ax, pos = plot_dataset(data, colors, ax=ax, add_colorbar=False)

        fig.savefig(output_path / f"groundtruth.svg")
        plt.close(fig)

    plot_soft_prediction_node(data, contracted_tree.root, id_node=0, cmap=cmap_heatmap, path=path, pos=pos)
    

def plot_soft_prediction_node(data, node, id_node, cmap, path, pos):

    if node.p is None:
        nb_points = get_nb_points(data)
        colors = cmap(np.ones(nb_points))
    else:
        colors = cmap(node.p)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
    plot_dataset(data, colors, ax=ax, cmap=cmap, pos=pos)
    fig.savefig(path / f"node_nb_{id_node:02d}.svg")
    plt.close(fig)

    if node.left_child is not None:
        id_left = get_next_id(id_node, 'left')
        plot_soft_prediction_node(data, node.left_child, id_left, cmap, path, pos=pos)
    if node.right_child is not None:
        id_right = get_next_id(id_node, 'right')
        plot_soft_prediction_node(data, node.right_child, id_right, cmap, path, pos=pos)


def plot_hard_predictions(data, ys_predicted, path=None):

    cmap_groundtruth = plt.cm.get_cmap('tab10')
    cmap_predictions = plt.cm.get_cmap('Set2')

    if path is not None:
        output_path = path
        output_path.mkdir(parents=True, exist_ok=True)

    if data['ys'] is not None:
        fig, (ax_true, ax_predicted) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        colors_true = labels_to_colors(data['ys'], cmap=cmap_groundtruth)
        ax_true = plot_dataset(data, colors_true, ax=ax_true, add_colorbar=False)
    else:
        fig, ax_predicted = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))

    colors_predicted = labels_to_colors(ys_predicted, cmap=cmap_predictions)
    ax_predicted = plot_dataset(data, colors_predicted, ax=ax_predicted, add_colorbar=False)

    fig.savefig(output_path / f"hard_clustering.svg")
    plt.close(fig)


def get_position(G, ys):
    if ys is not None:
        pos = nx.random_layout(G)
        ncls = np.max(ys) + 1
        xoff = np.sin(2 * np.pi * ys / ncls) * 2
        yoff = np.cos(2 * np.pi * ys / ncls) * 2
        for v in G:
            pos[v][0] += xoff[v]
            pos[v][1] += yoff[v]
        pos = nx.spring_layout(G, pos=pos, iterations=1)
    elif nx.is_connected(G):
        pos = nx.spectral_layout(G)
        pos = nx.spring_layout(G, pos=pos, k=.5, iterations=100)
    else:
        pos = nx.kamada_kawai_layout(G)
        pos = nx.spring_layout(G, pos=pos, k=.5, iterations=100)
    return pos


def plot_cuts(xs, ys, cuts, orders, path):

    path.mkdir(parents=True, exist_ok=True)
    plt.style.use('ggplot')
    plt.ioff()
    cmap = plt.cm.get_cmap('tab10')
    if ys is not None:
        normalise_ys = mpl.colors.Normalize(vmin=0, vmax=np.max(ys))

    if path is not None:
        path_cuts = path / 'points_cuts'
        path_cuts.mkdir(parents=True, exist_ok=True)

    _, nb_points = cuts.shape
    xs_embedded = TSNE(n_components=2).fit_transform(xs)

    for i, cut in enumerate(cuts):

        fig, (ax_true, ax_cut) = plt.subplots(
            nrows=1, ncols=2, figsize=(15, 15))
        ax_true.axis('off'), ax_true.grid(
            b=None), ax_true.set_title("True clusters")
        ax_cut.axis('off'), ax_cut.grid(
            b=None), ax_cut.set_title(f"cut of order {orders[i]}")

        if ys is not None:
            for y in np.unique(ys):
                xs_current = xs_embedded[ys == y]
                color = cmap(normalise_ys(y))
                color = np.array(color).reshape((1, -1))
                label = f'cluster {y}'

                ax_true.scatter(xs_current[:, 0], xs_current[:, 1],
                                c=color, label=label)
            ax_true.legend()
        else:
            color = np.array(COLOR_SILVER_RGB).reshape((1, -1))
            ax_true.scatter(xs_embedded[:, 0], xs_embedded[:, 1],
                            c=color)

        colors = np.zeros((nb_points, 4), dtype=float)
        colors[~cut] = COLOR_SILVER_RGB
        colors[cut] = COLOR_INDIGO_RGB

        ax_cut.scatter(xs_embedded[:, 0], xs_embedded[:, 1], c=colors)

        if path is not None:
            plt.savefig(path_cuts / f"cut number {i}.svg")

        plt.close(fig)


def plot_graph_cuts(G, ys, cuts, orders, path):

    path.mkdir(parents=True, exist_ok=True)
    plt.style.use('ggplot')
    plt.ioff()

    if path is not None:
        path_cuts = path / 'graph_cuts'
        path_cuts.mkdir(parents=True, exist_ok=True)

    _, nb_points = cuts.shape

    pos = get_position(G, ys)

    for i, cut in enumerate(cuts):
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        ax.axis('off')
        ax.grid(b=None)

        ax.set_title(f"cut of order {orders[i]}")

        colors = np.zeros((nb_points, 4), dtype=float)
        colors[~cut] = COLOR_SILVER_RGB
        colors[cut] = COLOR_INDIGO_RGB

        nx.draw_networkx(G, pos=pos, ax=ax, node_color=colors, cmap='tab10',
                         edge_color=COLOR_SILVER)

        if path is not None:
            plt.savefig(path_cuts / f"cut number {i}.svg")

        plt.close(fig)


def plot_evaluation(evaluations, path):

    path.mkdir(parents=True, exist_ok=True)
    plt.style.use('ggplot')
    plt.ioff()

    for nb_blocks, p_evaluations in evaluations.items():
        fig, ax = plt.subplots(1, 1)
        for i, (p, q_evaluations) in enumerate(p_evaluations.items()):

            cmap = plt.cm.get_cmap('tab10')
            rgb_color = np.array(cmap(i)).reshape(1, -1)
            v_scores = []
            for _, evaluation in q_evaluations.items():
                v_scores.append(evaluation["v_measure_score"])

            qs = list(q_evaluations.keys())

            ax.scatter(qs, v_scores, c=rgb_color)
            ax.plot(qs, v_scores, c=rgb_color[0], label=f'p = {p}')

            ax.xaxis.set_ticks(qs)
            ax.yaxis.set_ticks(np.arange(0, 1.05, 0.05))

        ax.set_ylabel('V-measure')
        ax.set_xlabel('q')
        ax.set_title(f'Number of blocks = {nb_blocks}')
        ax.legend()

        plt.savefig(path / f"Number of blocks {nb_blocks}.svg")
        plt.close(fig)


def add_lines(values, ax, left=True):
    
    n, m = values.shape
    old_i = None
    for j in np.arange(m):
        for i in np.arange(n):
            if values[i, j] == True:
                
                if old_i != i:
                    if left:
                        line = [(j - 0.5, i + 0.5),
                                (j - 0.5, i - 0.5), 
                                (j + 0.5, i - 0.5)]
                    else:
                        line = [(j - 0.5, i - 1.5),
                                (j - 0.5, i - 0.5), 
                                (j + 0.5, i - 0.5)]
                else:
                    line = [(j - 0.5, i - 0.5), 
                            (j + 0.5, i - 0.5)]
                        
                    
                path = patches.Polygon(line, facecolor='none', edgecolor='red',
                                       linewidth=2, closed=False, joinstyle='round')
                ax.add_patch(path)
                old_i = i
                break


def make_result_heatmap(data, ax, x_column, y_column, values_column):

    plt.rc('font', family='serif')

    df = data.pivot(index=y_column, columns=x_column, values=values_column
                   ).round(2).sort_index(ascending=False).sort_index(axis=1)
    xs = df.columns.to_numpy()
    ys = df.index.to_numpy()
    values = df.to_numpy()
    im = ax.imshow(values, cmap=plt.cm.get_cmap('Blues'), aspect='auto')

    ax.set_xticks(np.arange(len(xs)))
    ax.set_xticklabels(xs)
    ax.set_xlabel(x_column, fontsize=10, labelpad=10)

    ax.set_yticks(np.arange(len(ys)))
    ax.set_yticklabels(ys)
    ax.set_ylabel(y_column, rotation=0, fontsize=10, labelpad=15)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, 
             ha="center", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(ys)):
        for j in range(len(xs)):
            text = ax.text(j, i, values[i, j],
                           ha="center", va="center", color="black")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(len(xs)+1)-.5, minor=True)
    ax.set_yticks(np.arange(len(ys)+1)-.5, minor=True)
    ax.grid(b=True, which="minor", color="w", linestyle='-', linewidth=5)
    ax.tick_params(which="minor", bottom=False, left=False)


def make_benchmark_heatmap(exp_df, ref_df, x_column, y_column, values_column):

    plt.rc('font', family='serif')
    fig, ax = plt.subplots(figsize=(15, 10))

    exp = exp_df.pivot(index=y_column, columns=x_column, values=values_column
                      ).round(2).sort_index(ascending=False).sort_index(axis=1)
    ref = ref_df.pivot(index=y_column, columns=x_column, values=values_column
                      ).round(2).sort_index(ascending=False).sort_index(axis=1)

    difference_df = exp - ref

    xs = difference_df.columns.to_numpy()
    ys = difference_df.index.to_numpy()
    values = difference_df.to_numpy().round(2)
    
    heat_map = sns.heatmap(difference_df, center=0, cmap='BrBG', annot=True, linewidths=1, cbar_kws={'label': 'Difference in Rand score'})
    heat_map.set_yticklabels(heat_map.get_yticklabels(), rotation=0) 
    heat_map.set_ylabel(heat_map.get_ylabel(), rotation=0, labelpad=15) 

    return fig, ax