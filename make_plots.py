from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

nbs_blocks = [2, 3, 4, 5, 6, 7]
measures = ['completeness', 'homogeneity', 'v_measure_score']

plt.style.use('ggplot')
plt.ioff()
cmap = plt.cm.get_cmap('tab20')
path_plots = Path('plots')
path_plots.mkdir(parents=True, exist_ok=True)

for nb_blocks in nbs_blocks:
    for measure in measures:

        path = Path(f'data/nb_blocks_{nb_blocks}_{measure}.csv')
        if path.exists():

            fig, ax = plt.subplots(1, 1, figsize=(20, 10))

            df = pd.read_csv(path, header=0, index_col=0)
            average = df.reset_index().pivot_table(columns=["index"]).T

            qs = average.columns.to_numpy()
            ps = average.index.to_numpy()

            for i, (_, row) in enumerate(average.iterrows()):
                rgb_color = np.array(cmap(i)).reshape(1, -1)

                values = row.to_numpy()

                ax.scatter(qs, values, c=rgb_color)
                ax.plot(qs, values, c=rgb_color[0], label=f'p = {ps[i]}')

            ax.xaxis.set_ticks(qs)
            if measure != 'unassigned':
                ax.yaxis.set_ticks(np.arange(0, 1.05, 0.05))

            ax.set_ylabel(f'{measure}')
            ax.set_xlabel('q')
            ax.set_title(f'Number of blocks = {nb_blocks}')
            ax.legend()

            plt.savefig(path_plots / f"Number of blocks {nb_blocks} - {measure}.svg")
            plt.close(fig)
