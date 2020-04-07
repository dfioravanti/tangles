import numpy as np
import networkx as nx

from matplotlib import pyplot as plt

COLOR_SILVER = '#C2C2C2'


def plot_predictions(xs, ys, predictions_of_order, path=None):

    """
    For each tangle print a heatmap that shows how many cuts each point satisfies.

    Parameters
    ----------
    all_cuts: array of shape [nb_cuts, nb_points]
        the collection of all cuts
    ys: array of shape [n_points]
        The array of class labels
    predictions_of_order: dict of list of Specification
        A dictionary where the key is the order and the value is a list of all the tangles of that order
    path:

    Returns
    -------

    """

    plt.style.use('ggplot')
    plt.ioff()
    cmap = plt.cm.get_cmap('tab10')

    if path is not None:
        output_path = path / 'points prediction'
        output_path.mkdir(parents=True, exist_ok=True)

    for order, prediction in predictions_of_order.items():

        f, (ax_true, ax_pred) = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))
        ax_true.axis('off'), ax_true.grid(b=None), ax_true.set_title("True clusters")
        ax_pred.axis('off'), ax_pred.grid(b=None), ax_pred.set_title("Predicted clusters")

        ax_true.scatter(xs[:, 0], xs[:, 1], c=ys, cmap=cmap)
        ax_pred.scatter(xs[:, 0], xs[:, 1], c=prediction, cmap=cmap)

        if path is None:
            plt.show()
        else:
            plt.savefig(output_path / f"Tangle order {order}.svg")

        plt.close(f)


def plot_predictions_graph(G, ys, predictions_of_order, path=None):

    """
    For each tangle print a heatmap that shows how many cuts each point satisfies.

    Parameters
    ----------
    all_cuts: array of shape [nb_cuts, nb_points]
        the collection of all cuts
    ys: array of shape [n_points]
        The array of class labels
    predictions_of_order: dict of list of Specification
        A dictionary where the key is the order and the value is a list of all the tangles of that order
    path:

    Returns
    -------

    """

    plt.style.use('ggplot')
    plt.ioff()

    if path is not None:
        output_path = path / 'graph prediction'
        output_path.mkdir(parents=True, exist_ok=True)

    pos = nx.spectral_layout(G)
    pos = nx.spring_layout(G, pos=pos, k=.5, iterations=100)

    cmap = plt.cm.get_cmap('tab10')

    for order, prediction in predictions_of_order.items():

        f, (ax_true, ax_pred) = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))
        ax_true.axis('off'), ax_true.grid(b=None), ax_true.set_title("True clusters")
        ax_pred.axis('off'), ax_pred.grid(b=None), ax_pred.set_title("Predicted clusters")

        nx.draw_networkx(G, pos=pos, ax=ax_true, node_color=ys,
                         cmap=cmap, edge_color=COLOR_SILVER)

        nx.draw_networkx(G, pos=pos, ax=ax_pred, node_color=prediction,
                         cmap=cmap, edge_color=COLOR_SILVER)

        if path is None:
            plt.show()
        else:
            plt.savefig(output_path / f"Tangle order {order}.svg")

        plt.close(f)


def plot_graph_cuts(xs, cuts, orders, type, path):
    plt.style.use('ggplot')
    plt.ioff()

    if path is not None:
        path_cuts = path / 'graph_cuts'
        path_cuts.mkdir(parents=True, exist_ok=True)

    _, nb_points = cuts.shape

    if type == 'graph':
        pos = nx.spectral_layout(xs)
        pos = nx.spring_layout(xs, pos=pos, k=.5, iterations=100)

    for i, cut in enumerate(cuts):
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        ax.axis('off')
        ax.grid(b=None)

        ax.set_title(f"cut of order {orders[i]}")

        colors = np.zeros(nb_points)
        colors[cut] = (i % 9) + 1

        if type == 'graph':
            nx.draw_networkx(xs, pos=pos, ax=ax, node_color=colors, cmap='tab10',
                             edge_color=COLOR_SILVER)

        if path is not None:
            plt.savefig(path_cuts / f"cut number {i}.svg")

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
