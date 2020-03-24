import numpy as np
import networkx as nx
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE

from matplotlib import pyplot as plt


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

    _, nb_points = all_cuts.shape
    nb_classes = max(ys) + 1

    plt.style.use('ggplot')
    for order, tangles in tangles_by_orders.items():

        nb_tangles = len(tangles)
        nrows = (nb_tangles // 3) + 1

        f, axs = plt.subplots(nrows=nrows, ncols=3, figsize=(12, 12))

        if nb_tangles == 1:
            axs = np.array(axs)
        axs = axs.flatten()

        for i, tangle in enumerate(tangles):
            idx = list(tangle.specification.keys())
            orientations = np.array(list(tangle.specification.values()), dtype=bool)
            matching_cuts = np.sum((all_cuts[idx, :].T == orientations), axis=1)

            axs[i].scatter(np.arange(1, nb_points + 1), matching_cuts, c=ys)
            axs[i].axis('off')
            axs[i].set_title(f"Tangle number {i} ({len(idx)} cuts)")
            for j in range(1, nb_classes):
                nb_in_class = np.sum(ys == j-1)
                axs[i].vlines(x=j * nb_in_class + 0.5, ymin=0, ymax=max(matching_cuts),
                              linestyles='dashed')

            if path is None:
                plt.show()
            else:
                plt.savefig(f"Tangle order {order}.png")

        plt.close(f)


def plot_heatmap_graph(G, all_cuts, tangles_by_orders, path=None):

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

    A = nx.to_numpy_array(G)
    nb_vertex = len(A)
    pos = nx.spring_layout(G, k=.5, iterations=100)

    #pos = np.random.rand(nb_vertex, 2)
    #pos[round(nb_vertex/2):, :] += 1

    for order, tangles in tangles_by_orders.items():

        nb_tangles = len(tangles)
        nrows = nb_tangles // 2 + nb_tangles % 2
        ncols = 2 if nb_tangles >= 2 else 1

        f, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 12))

        if nb_tangles == 1:
            axs = np.array(axs)
        axs = axs.flatten()

        for i, tangle in enumerate(tangles):
            idx = list(tangle.specification.keys())
            orientations = np.array(list(tangle.specification.values()), dtype=bool)
            matching_cuts = np.sum((all_cuts[idx, :].T == orientations), axis=1)

            cmap = plt.cm.get_cmap('tab10')
            my_cmap = cmap(i)
            my_cmap = np.array([my_cmap])
            my_cmap = np.tile(my_cmap, (nb_vertex, 1))
            my_cmap[:, -1] = np.linspace(0.1, 1, nb_vertex)
            my_cmap = ListedColormap(my_cmap)

            nx.draw_networkx(G, pos=pos, ax=axs[i], node_color=matching_cuts, cmap=my_cmap)

            if path is None:
                plt.show()
            else:
                plt.savefig(path / f"Tangle order {order}.png")

        plt.close(f)


def plot_cuts(xs, cuts, orders, type, path):
    plt.style.use('ggplot')
    plt.ioff()

    _, nb_points = cuts.shape

    if type == 'graph':
        pos = nx.spring_layout(xs, k=.5, iterations=100)

    for i, cut in enumerate(cuts):
        fig, ax = plt.subplots(1, 1)

        ax.set_title(f"cut of order {orders[i]}")

        colors = np.zeros(nb_points)
        colors[cut] = (i % 9) + 1

        if type == 'graph':
            nx.draw_networkx(xs, pos=pos, ax=ax, node_color=colors, cmap='tab10')

        if path is not None:
            plt.savefig(path / f"cut number {i}.png")

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


