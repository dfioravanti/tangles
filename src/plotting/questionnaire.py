import numpy as np

from sklearn.manifold import TSNE

from matplotlib import pyplot as plt
import seaborn as sns


def plot_questionnaire(xs, ys, cs=None):

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
    f, axes = plt.subplots(1, 1, figsize=(7, 7))
    sns.despine(left=True, bottom=True, right=True)

    n_classes = np.max(ys) + 1
    palette = sns.color_palette("colorblind", n_classes)
    sns.scatterplot(xs_embedded[:, 0], xs_embedded[:, 1],
                    hue=ys, legend='full', palette=palette)

    if cs is not None:
        ys_cs = np.arange(len(cs))
        sns.scatterplot(cs_embedded[:, 0], cs_embedded[:, 1], marker='x',
                        hue=ys_cs, legend=False, palette=palette)

    plt.setp(axes, yticks=[], xticks=[])
    plt.legend(bbox_to_anchor=(1.01, 0.5), loc="center left", borderaxespad=0.)
    plt.show()


if __name__ == '__main__':

    from src.datasets.questionnaire import make_synthetic_questionnaire

    xs, ys, cs = make_synthetic_questionnaire(n_samples=500, n_features=32, n_mindsets=10, tolerance=0.65)
    plot_questionnaire(xs, ys, cs)
