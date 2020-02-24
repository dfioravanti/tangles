import numpy as np

from sklearn.manifold import TSNE

from matplotlib import pyplot as plt
import seaborn as sns


def plot_questionnaire(xs, ys, cs=None, ax=None):

    """

    Plots a questionnaire using TSNE with the hamming distance.
    If the centers are passed they are printed with a different marker.

    Parameters
    ----------

    xs : array of shape [n_samples, n_features]
        The generated samples.
    ys : array of shape [n_samples]
        The integer labels for mindset membership of each sample.
    cs : array of shape [n_mindsets, n_features], optional (default=None)
        The coordinates of the centers of the mindsets
    ax
    Returns
    -------

    """

    tsne = TSNE(metric='manhattan')

    if cs is not None:
        xs_to_reduce = np.concatenate([xs, cs])
    else:
        xs_to_reduce = xs

    xs_embedded = tsne.fit_transform(xs_to_reduce)
    if cs is not None:
        cs_embedded = xs_embedded[-len(cs):, :]
        xs_embedded = xs_embedded[:-len(cs), :]

    sns.set_style("white")

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    sns.despine(ax=ax, left=True, bottom=True, right=True)

    n_classes = np.max(ys) + 1
    palette = sns.color_palette("colorblind", n_classes)
    sns.scatterplot(xs_embedded[:, 0], xs_embedded[:, 1], ax=ax,
                    hue=ys, legend='full', palette=palette)

    if cs is not None:
        ys_cs = np.arange(len(cs))
        sns.scatterplot(cs_embedded[:, 0], cs_embedded[:, 1], ax=ax, marker='x',
                        hue=ys_cs, legend=False, palette=palette)

    ax.legend(bbox_to_anchor=(1.01, 0.5), loc="center left", borderaxespad=0.)

if __name__ == '__main__':

    from src.datasets.generation.questionnaire import make_synthetic_questionnaire

    n_samples = 100
    n_features = 15
    n_mindsets = 2
    tolerance = 0.6
    seed = 42

    xs, ys, cs = make_synthetic_questionnaire(n_samples, n_features, n_mindsets, tolerance, seed)
    plot_questionnaire(xs, ys, cs)
