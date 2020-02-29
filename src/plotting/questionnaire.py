import numpy as np

from sklearn.manifold import TSNE

from matplotlib import pyplot as plt
import matplotlib.cm as cm


def plot_tangles_on_questionnaire(xs, ys, masks_tangles, orders, path=None):

    tsne = TSNE(metric='manhattan')
    xs_embedded = tsne.fit_transform(xs)

    plt.style.use('ggplot')
    size_markers = 10

    n_columns = len(masks_tangles)
    f, axs = plt.subplots(1, n_columns + 1, figsize=(7, 7))
    for ax in axs:
        ax.set_axis_off()

    axs[0].set(title='True')
    scatter = axs[0].scatter(xs_embedded[:, 0], xs_embedded[:, 1], c=ys, s=size_markers, cmap='coolwarm')
    leg = axs[0].legend(*scatter.legend_elements(), title="Classes")
    axs[0].add_artist(leg)

    for i, mask in enumerate(masks_tangles, 1):

        ax = axs[i]
        ax.set(title=f"Tangle of order {orders[i-1]}")

        ys_pred = np.zeros_like(ys)
        for j, tangle in enumerate(mask, 1):
            ys_pred[tangle.reshape(-1)] = j

        scatter = ax.scatter(xs_embedded[:, 0], xs_embedded[:, 1], c=ys_pred, s=size_markers, cmap='coolwarm')
        leg = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(leg)

    if path is None:
        plt.show()
