import numpy as np
import networkx as nx
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE

from matplotlib import pyplot as plt

COLOR_SILVER = '#C2C2C2'


def plot_heatmap(all_cuts, ys, tangles_by_orders, path=None):

    """
    For each tangle print a heatmap that shows how many cuts each point satisfies.

    Parameters
    ----------
    all_cuts: array of shape [nb_cuts, nb_points]
        the collection of all cuts
    ys: array of shape [n_points]
        The array of class labels
    tangles_by_orders: dict of list of Specification
        A dictionary where the key is the order and the value is a list of all the tangles of that order
    path:

    Returns
    -------

    """
    plt.style.use('ggplot')
    plt.ioff()

    _, nb_points = all_cuts.shape
    nb_classes = max(ys) + 1

    for order, tangles in tangles_by_orders.items():

        nb_tangles = len(tangles)
        nrows = (nb_tangles // 3) + 1

        f, axs = plt.subplots(nrows=nrows, ncols=3, figsize=(12, 12))

        if nb_tangles == 1:
            axs = np.array(axs)
        axs = axs.flatten()
        for ax in axs:
            ax.axis('off')
            ax.grid(b=None)

        idx_in_all = cuts_in_all_tangles(tangles)

        for i, tangle in enumerate(tangles):

            cs, os = [], []

            for c, o in tangle.specification.items():
                if c not in idx_in_all:
                    cs.append(c)
                    os.append(o)

            os = np.array(os, dtype=bool)
            matching_cuts = np.sum((all_cuts[cs, :].T == os), axis=1)

            axs[i].scatter(np.arange(1, nb_points + 1), matching_cuts, c=ys)
            axs[i].axis('off')
            axs[i].set_title(f"Tangle number {i}")
            for j in range(1, nb_classes):
                nb_in_class = np.sum(ys == j-1)
                axs[i].vlines(x=j * nb_in_class + 0.5, ymin=0, ymax=max(matching_cuts),
                              linestyles='dashed')

            if path is None:
                plt.show()
            else:
                plt.savefig(path / f"Tangle order {order}.png")

        plt.close(f)


def cuts_in_all_tangles(tangles):

    idx_cuts = set()
    for tangle in tangles:
        idx_cuts = idx_cuts.union(set(tangle.specification.keys()))

    idx_in_all = []
    for idx_cut in idx_cuts:
        current_o = None

        in_all = True
        for tangle in tangles:
            idx = list(tangle.specification.keys())
            orr = list(tangle.specification.values())
            try:
                i = idx.index(idx_cut)
                o = orr[i]
                if current_o is None:
                    current_o = o
                elif o != current_o:
                    in_all = False
                    break
            except ValueError:
                in_all = False
                break

        if in_all:
            idx_in_all.append(idx_cut)

    return idx_in_all


def plot_heatmap_graph(G, all_cuts, predictions, path=None):

    """
    For each tangle print a heatmap that shows how many cuts each point satisfies.

    Parameters
    ----------
    all_cuts: array of shape [nb_cuts, nb_points]
        the collection of all cuts
    ys: array of shape [n_points]
        The array of class labels
    predictions: dict of list of Specification
        A dictionary where the key is the order and the value is a list of all the tangles of that order
    path:

    Returns
    -------

    """

    plt.style.use('ggplot')
    plt.ioff()

    pos = nx.spectral_layout(G)
    pos = nx.spring_layout(G, pos=pos, k=.5, iterations=100)

    for order, prediction in predictions.items():

        f, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

        ax.axis('off')
        ax.grid(b=None)
        cmap = plt.cm.get_cmap('tab10')

        nx.draw_networkx(G, pos=pos, ax=ax, node_color=prediction,
                         cmap=cmap, edge_color=COLOR_SILVER)

        if path is None:
            plt.show()
        else:
            plt.savefig(path / f"Tangle order {order}.svg")

        plt.close(f)


def plot_cuts(xs, cuts, orders, type, path):
    plt.style.use('ggplot')
    plt.ioff()

    _, nb_points = cuts.shape

    if type == 'graph':
        pos = nx.spectral_layout(xs)
        pos = nx.spring_layout(xs, pos=pos, k=.5, iterations=100)

    for i, cut in enumerate(cuts):
        fig, ax = plt.subplots(1, 1)
        ax.axis('off')
        ax.grid(b=None)

        ax.set_title(f"cut of order {orders[i]}")

        colors = np.zeros(nb_points)
        colors[cut] = (i % 9) + 1

        if type == 'graph':
            nx.draw_networkx(xs, pos=pos, ax=ax, node_color=colors, cmap='tab10',
                             edge_color=COLOR_SILVER)

        if path is not None:
            plt.savefig(path / f"cut number {i}.svg")

        plt.close(fig)


def plot_evaluation(evaluations, path):

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


# Old code

def plot_dataset(xs, ys, path=None):

    tsne = TSNE(metric='manhattan')
    xs_embedded = tsne.fit_transform(xs)

    plt.style.use('ggplot')
    size_markers = 10

    f, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.set_axis_off()

    ax.set(title='True')
    scatter = ax.scatter(xs_embedded[:, 0], xs_embedded[:, 1], c=ys, s=size_markers, cmap='coolwarm')
    leg = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(leg)

    path = path / "dataset.svg"

    if path is None:
        plt.show()
    else:
        plt.savefig(path)


