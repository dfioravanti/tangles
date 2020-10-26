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

df_mean = pd.read_csv('{}/cost_sum_sbm.csv'.format(root))

cost_mean = df_mean.cost

quality_mean = df_mean.quality

fig, ax1 = plt.subplots(1, 1, figsize=(singlecol_width, 3))

ax1.scatter(cost_mean, quality_mean, marker='.', s=1)
ax1.set_ylim(0, 1)
ax1.grid('on', alpha=0.5)
ax1.set_xlabel('Not normalized cost')
ax1.set_ylabel('Normalized mutual information')

plt.tight_layout()

plt.savefig("{}/cost_function_sbm_sum.pdf".format(root))
plt.close()