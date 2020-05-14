import numpy as np
import networkx as nx

import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.patches as patches

from sklearn.manifold import TSNE

import altair as alt
from altair.expr import datum

from src.config import NAN
from src.utils import get_points_to_plot


# Standard colors for uniform plots
COLOR_SILVER = '#C0C0C0'
COLOR_SILVER_RGB = (192 / 255, 192 / 255, 192 / 255) + (0.2,)
COLOR_INDIGO_RGB = (55 / 255, 0 / 255, 175 / 255) + (0.5,)
COLOR_CARNATION_RGB = np.array((247 / 255, 96 / 255, 114 / 255, 1)).reshape((1, -1))
CMAP = plt.cm.get_cmap('Blues')

# TODO: Fix the comments in this file

def plot_heatmap(data, tangles, cuts, path=None):

    if data['xs'] is not None:
        plot_heatmap_points(xs=data['xs'], ys=data['ys'], cs=data['cs'],
                           tangles=tangles, cuts=cuts, path=path)


def plot_heatmap_points(xs, ys, cs, tangles, cuts, path=None):
    """
    For each tangle print a heatmap that shows how many cuts each point satisfies.

    Parameters
    ----------
    
    Returns
    -------

    """

    if path is not None:
        output_path = path
        output_path.mkdir(parents=True, exist_ok=True)

    plt.style.use('ggplot')
    plt.ioff()

    mpl.rcParams.update({'font.size': 8})
   
    nb_points = len(xs)
    xs_embedded, cs_embedded = get_points_to_plot(xs, cs)
    if ys is not None:
        ys_normalized = mpl.colors.Normalize(vmin=0, vmax=np.max(ys))

    color_tangles = mpl.colors.Normalize(vmin=0, vmax=np.max(len(tangles)))
    f, axs = plt.subplots(nrows=1, ncols=len(tangles), figsize=(20, 10), )

    for i, tangle in enumerate(tangles):       
        ax = axs[i]
        ax.grid(b=None)
        ax.set_title(f'Tangle number {i+1}')
        ax.set_xticks([], [])
        ax.set_yticks([], [])

        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    
        ax.set_facecolor(CMAP(0))

        matching_cuts = np.zeros(nb_points, dtype='int')
        for cut, orr in tangle.specification.items():
            if np.sum(cuts[cut] == orr) != nb_points:
                matching_cuts = matching_cuts + (cuts[cut] == orr)
        
        ax.hexbin(xs_embedded[:, 0], xs_embedded[:, 1], 
                  C=matching_cuts, cmap=CMAP, bins='log', gridsize=25)

        if cs is not None:
            ax.scatter(cs_embedded[:, 0], cs_embedded[:, 1], c=COLOR_CARNATION_RGB, marker='o', s=10)

    f.tight_layout()

    if path is None:
        plt.show()
    else:
        plt.savefig(output_path / f"Best heatmap.svg")

    plt.close(f)


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

    path.mkdir(parents=True, exist_ok=True)
    plt.style.use('ggplot')
    plt.ioff()
    cmap = plt.cm.get_cmap('tab20')

    if ys is not None:
        normalise_ys = mpl.colors.Normalize(vmin=0, vmax=np.max(ys))

    xs_embedded = TSNE(n_components=2).fit_transform(xs)

    if path is not None:
        output_path = path / 'points prediction'
        output_path.mkdir(parents=True, exist_ok=True)

    for order, prediction in predictions_of_order.items():

        normalise_pred = mpl.colors.Normalize(vmin=0, vmax=np.max(prediction))

        f, (ax_true, ax_pred) = plt.subplots(
            nrows=1, ncols=2, figsize=(15, 15))
        ax_true.axis('off'), ax_true.grid(
            b=None), ax_true.set_title("True clusters")
        ax_pred.axis('off'), ax_pred.grid(
            b=None), ax_pred.set_title("Predicted clusters")

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

        for y in np.unique(prediction):

            xs_current = xs_embedded[prediction == y]
            if y != NAN:
                color = cmap(normalise_pred(y))
                label = f'cluster {y}'
            else:
                color = COLOR_SILVER_RGB
                label = f'no cluster'

            color = np.array(color).reshape((1, -1))
            ax_pred.scatter(xs_current[:, 0], xs_current[:, 1],
                            c=color, label=label)

        ax_pred.legend()

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

    path.mkdir(parents=True, exist_ok=True)
    plt.style.use('ggplot')
    plt.ioff()
    cmap = plt.cm.get_cmap('tab10')
    if ys is not None:
        normalise_ys = mpl.colors.Normalize(vmin=0, vmax=np.max(ys))

    if path is not None:
        output_path = path / 'graph prediction'
        output_path.mkdir(parents=True, exist_ok=True)

    pos = get_position(G, ys)

    for order, prediction in predictions_of_order.items():

        normalise_pred = mpl.colors.Normalize(vmin=0, vmax=np.max(prediction))

        f, (ax_true, ax_pred) = plt.subplots(
            nrows=1, ncols=2, figsize=(15, 15))
        ax_true.axis('off'), ax_true.grid(
            b=None), ax_true.set_title("True clusters")
        ax_pred.axis('off'), ax_pred.grid(
            b=None), ax_pred.set_title("Predicted clusters")

        colors = np.zeros((len(ys), 4))
        if ys is not None:
            for y in np.unique(ys):
                color = cmap(normalise_ys(y))
                colors[ys == y, :] = color
        else:
            colors[:] = COLOR_SILVER_RGB

        nx.draw_networkx(G, pos=pos, ax=ax_true,
                         node_color=colors,
                         edge_color=COLOR_SILVER)

        colors = np.zeros((len(prediction), 4))
        for y in np.unique(prediction):

            if y != NAN:
                color = cmap(normalise_pred(y))
                colors[prediction == y, :] = color
            else:
                colors[prediction == y, :] = COLOR_SILVER_RGB

        nx.draw_networkx(G, pos=pos, ax=ax_pred,
                         node_color=colors,
                         edge_color=COLOR_SILVER)

        if path is None:
            plt.show()
        else:
            plt.savefig(output_path / f"Tangle order {order}.svg")

        plt.close(f)


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

# Plots results


def make_homogeneity_plot(df, title, x_axis, y_axis, facet_on):

    chart = alt.Chart(df, width=800, height=300).mark_rect().encode(
        alt.X(x_axis, type='ordinal', sort=alt.EncodingSortField(
            field=x_axis, order='ascending',), axis=alt.Axis(grid=True)),
        alt.Y(y_axis, type='ordinal', sort=alt.EncodingSortField(
            field=y_axis, order='descending'), axis=alt.Axis(grid=True)),
        alt.Color('homogeneity', type='quantitative',
                  title='Homogeneity', scale=alt.Scale(domain=[0, 1])),
    ).properties(
        title=title
    )

    return chart


def make_full_plot(df, title, x_axis, y_axis, facet_on):

    v_measure_chart = make_homogeneity_plot(
        df, title, x_axis, y_axis, facet_on)

    text = alt.Chart(df, width=800, height=300).mark_text().encode(
        alt.X(x_axis, type='ordinal', sort=alt.EncodingSortField(
            field=x_axis, order='ascending'), axis=alt.Axis(grid=True)),
        alt.Y(y_axis, type='ordinal', sort=alt.EncodingSortField(
            field=y_axis, order='descending'), axis=alt.Axis(grid=True)),
        alt.Text('order'),
    ).facet(
        facet=alt.Facet(facet_on, type='nominal', title=None),
        title='best_order/max_order'
    )

    chart = alt.vconcat(v_measure_chart, text)
    chart = graphic_settings(chart)

    return chart


def graphic_settings(chart):
    chart = chart.configure_title(
        fontSize=14,
        font='Courier',
        anchor='middle',
        color='gray'
    ).configure_axis(
        gridOpacity=0.0,

        labelFont='Courier',
        labelColor='black',

        titleFont='Courier',
        titleColor='gray',
        grid=False
    ).configure_axisX(
        labelAngle=0,
    ).configure_legend(
        labelFont='Courier',
        labelColor='black',

        titleFont='Courier',
        titleColor='gray',
        titleAnchor='middle'
    ).configure_view(strokeOpacity=0)

    return chart

def add_lines(values, ax, left=True):
    
    n, m = values.shape
    old_i = None
    for j in np.arange(m):
        for i in np.arange(n):
            if values[i, j] == True:
                
                if old_i != i and old_i is not None:
                    if left:
                        line = [(j - 0.5, i + 0.5),
                                (j - 0.5, i - 0.5), 
                                (j + 0.5, i - 0.5)]
                    else:
                        line = [(j - 0.5, i - 1.5),
                                (j - 0.5, i - 0.5), 
                                (j + 0.5, i - 0.5)]
                else:
                        line = [
                                (j - 0.5, i - 0.5), 
                                (j + 0.5, i - 0.5)]
                        
                    
                path = patches.Polygon(line, facecolor='none', edgecolor='red',
                                       linewidth=2, closed=False, joinstyle='round')
                ax.add_patch(path)
                old_i = i
                break


def make_result_heatmap(data, title, ax, x_column, y_column, values_column):

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
    ax.set_title(title, fontsize=20, pad=10)
