import matplotlib
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

from src.utils import normalize

singlecol_width = 3.2621875
doublecol_width = 6.7753125

SMALL_SIZE = 9
NORMAL_SIZE = 10
LARGE_SIZE = 12

plt.rcParams['font.size'] = NORMAL_SIZE
plt.rcParams['axes.labelsize'] = SMALL_SIZE
plt.rcParams['xtick.labelsize'] = SMALL_SIZE
plt.rcParams['ytick.labelsize'] = SMALL_SIZE
plt.rcParams['legend.fontsize'] = SMALL_SIZE
plt.rcParams['figure.titlesize'] = SMALL_SIZE
plt.rcParams['axes.axisbelow'] = True

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

cmap = matplotlib.cm.get_cmap('Paired')

COLOR1 = cmap(9)
COLOR2 = cmap(6)
COLOR3 = cmap(3)

root = os.path.abspath(os.curdir)

df_mean = pd.read_csv('{}/cost_mean.csv'.format(root))

cost_mean = df_mean.cost

quality_mean = df_mean.quality

fig, ax1 = plt.subplots(1, 1, figsize=(singlecol_width, 3))

ax1.scatter(np.zeros(5), cost_mean[:5], marker='o', color=COLOR1, label='different')
ax1.scatter(np.zeros(5), cost_mean[5:10], marker='x', color=COLOR3, label='same')
ax1.scatter(np.zeros(5), cost_mean[10:], marker='s', color=COLOR2, label='random')

#ax1.set_yscale('log')
#ax1.set_ylim(10**-6, 10**0)

df_sum = pd.read_csv('{}/cost_sum.csv'.format(root))

cost_sum = df_sum.cost

quality_sum = df_sum.quality

ax2 = ax1.twinx()

ax2.scatter(np.ones(5), cost_sum[:5], marker='o', color=COLOR1)
ax2.scatter(np.ones(5), cost_sum[5:10], marker='x', color=COLOR3)
ax2.scatter(np.ones(5), cost_sum[10:], marker='s', color=COLOR2)


locs, labels = plt.xticks()
plt.xticks([0, 1], ['Normalized', 'Not Normalized'])

ax1.set_ylabel("cost")

ax1.grid('on', axis='x', linewidth=0.25, alpha=0.5)

ax1.set_xlim(-0.5, 1.5)

ax1.legend(ncol=3, framealpha=1, labelspacing=0, handleheight=1, fontsize='small', bbox_to_anchor=(0.5, 1), loc='lower center')

plt.tight_layout()

plt.savefig("{}/cost_function.pdf".format(root))
plt.close()