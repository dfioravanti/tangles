import numpy as np

from sklearn.manifold import TSNE

from matplotlib import pyplot as plt
from matplotlib import cm


def plot_tangles_on_questionnaire(xs, ys, predictions, path=None):

    tsne = TSNE(metric='manhattan')
    xs_embedded = tsne.fit_transform(xs)

    plt.style.use('ggplot')
    size_markers = 10

    n_columns = len(predictions)
    f, axs = plt.subplots(1, n_columns + 1, figsize=(7, 7))
    for ax in axs:
        ax.set_axis_off()

    axs[0].set(title='True')
    scatter = axs[0].scatter(xs_embedded[:, 0], xs_embedded[:, 1], c=ys, s=size_markers, cmap='coolwarm')
    leg = axs[0].legend(*scatter.legend_elements(), title="Classes")
    axs[0].add_artist(leg)

    i_ax = 1
    for order, prediction in predictions.items():

        ax = axs[i_ax]
        ax.set(title=f"Tangle of order {order}")
        nb_clusters = np.max(prediction) + 1
        colors = [cm.coolwarm(x) for x in np.linspace(0, 1, nb_clusters)]

        for cluster, color in zip(range(nb_clusters), colors):
            mask = (prediction == cluster)
            xs, ys = xs_embedded[mask, 0], xs_embedded[mask, 1]
            label = "No tangle" if cluster == 0 else f"tangle {cluster}"

            ax.scatter(xs, ys, color=color,
                       s=size_markers, cmap='coolwarm',
                       label=label)

        ax.legend()
        i_ax += 1

    if path is None:
        plt.show()
