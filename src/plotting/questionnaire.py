import numpy as np

from sklearn.manifold import TSNE

from matplotlib import pyplot as plt
import seaborn as sns


def plot_tangles_on_questionnaire(xs, ys, masks_tangles, orders, path=None):

    tsne = TSNE(metric='manhattan')
    xs_embedded = tsne.fit_transform(xs)

    sns.set_style("white")
    n_columns = len(masks_tangles)
    f, axs = plt.subplots(1, n_columns+1, figsize=(7, 7))

    for ax in axs:
        sns.despine(ax=ax, left=True, bottom=True, right=True)
        ax.set_axis_off()

    n_classes = np.max(ys) + 1
    palette = sns.color_palette("colorblind", n_classes)
    axs[0].set(title='True')
    sns.scatterplot(xs_embedded[:, 0], xs_embedded[:, 1], ax=axs[0],
                    hue=ys, legend='full', palette=palette)

    print(len(masks_tangles))
    for i, mask in enumerate(masks_tangles):
        hue = np.zeros_like(ys)
        n_classes = len(mask)
        palette = sns.color_palette("colorblind", n_classes)
        for j, tangle in enumerate(mask):
            hue[tangle.reshape(-1)] = j

        ax = axs[i + 1]
        ax.set(title=f"Tangle of order {orders[i]}")
        sns.scatterplot(xs_embedded[:, 0], xs_embedded[:, 1], ax=ax,
                        hue=hue, legend='full', palette=palette)

    if path is None:
        plt.show()
