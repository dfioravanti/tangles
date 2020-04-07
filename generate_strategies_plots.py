import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import copy
from multiprocessing import Pool
from bokeh.plotting import figure, show, output_file

from src.loading import get_dataset_and_order_function
from src.execution import compute_cuts, compute_tangles, order_cuts, compute_clusters, compute_evaluation
from src.config import load_validate_settings, VALID_PREPROCESSING

def onerun(args):
    print(args.dataset.sbm)
    
    print("Load data\n", flush=True)
    xs, ys, G, order_function = get_dataset_and_order_function(args.dataset, args.seed)

    cuts = dict()
    orders = dict()
    times = dict()

    names = ['kneip', 'local_min', 'kernighan_lin', 'local_min_bounded', 'random']
    for name in names:
        print(f"Find cuts using {name}", flush=True)
        if name == 'kernighan_lin':
            args.preprocessing.name = ['karnig_lin']
        else:
            args.preprocessing.name = [name]
        start = time.time()
        all_cuts = compute_cuts(xs.copy(), args.preprocessing)

        print("Compute order", flush=True)
        all_cuts, all_orders = order_cuts(all_cuts, order_function)

        max_order = np.int(np.ceil(np.max(all_orders)))
        min_order = np.int(np.floor(np.min(all_orders)))
        print(f"\tOrder range: {min_order}--{max_order} \n", flush=True)

        cuts[name] = all_cuts
        orders[name] = all_orders
        times[name] = time.time() - start
    return (args, G, cuts, orders, times)

if __name__ == "__main__":
    baseargs = load_validate_settings('./')
    baseargs.dataset.name = 'sbm'

    allargs = []
    for N in [50, 100, 150, 200]:
        for p in [.1, .6, .25, .4]:
            for nbl in [2, 5, 3, 8]:
                for q in [p/3, p/10, p/5, p/7]:
                    newargs = copy.deepcopy(baseargs)
                    bsize = N // nbl
                    newargs.dataset.sbm.nb_blocks = nbl
                    newargs.dataset.sbm.p = p
                    newargs.dataset.sbm.q = q
                    newargs.dataset.sbm.block_size = bsize

                    allargs.append(newargs)


    with Pool() as pool:
        for (args, G, cuts, orders, times) in pool.imap_unordered(onerun, allargs, 1):
            N = len(G)

            fig, ax = plt.subplots(figsize=(25,15))
            cmap = plt.cm.get_cmap('Set1')

            for i, name in enumerate(cuts):
                balance = 0.5 - np.abs(cuts[name].sum(axis=1) / N - 0.5)
                order = orders[name] / len(G.edges)
                xfuzz = (np.random.rand(*balance.shape) - .5) * (1 / len(G))
                yfuzz = (np.random.rand(*balance.shape) - .5) * 0.001

                color = np.array(cmap(i))

                ax.plot(balance + xfuzz, order + yfuzz, marker='.', ms=10, linestyle='', label=f'{name} ($m={len(order)}$, took {times[name] :.1f}s)', color=color)
            ax.set(xlabel='Cut balance', ylabel='Cut value in $|E|$')

            nb, p, q = args.dataset.sbm.nb_blocks, args.dataset.sbm.p, args.dataset.sbm.q

            ax.hlines([(i*N/nb) * (N - i*N/nb) * q / len(G.edges) for i in range(1, 1 + nb//2)], 0, 0.5, linestyles='dashed', label='expected values of blockwise cuts')

            plt.setp(ax, xticks=[i / nb for i in range(1 + nb // 2)] + ([0.5] if nb % 2 == 1 else []),
                    xticklabels=[f'${i}/{nb}$' for i in range(1 + nb // 2)] + (['1/2'] if nb % 2 == 1 else []),
                    xlim=(0, 0.51), ylim=(0,0.51))

            ax.grid(True)
            ax.legend()
            if args.dataset.name == 'sbm':
                ax.set_title(f'Comparison of sampling strategies, SBM with {nb} blocks of {N//nb} nodes, p = {p}, q = p/{np.round(p/q) :.0f} \n (Data is fuzzed for readability)')
                fig.savefig(f'sampling/sbm{nb}blocks{50 * np.round(N/50) :.0f}nodes{p}p{np.round(p/q) :.0f}factor.png')
            plt.close(fig)
