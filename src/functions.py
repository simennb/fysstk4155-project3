import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def plot_heatmap(x, y, z, xlab, ylab, zlab, title, save, fig_path, task=None, fs=14, xt='int', yt='int'):
    ticks = {'int': '%d', 'exp': '%1.1e', 'str': '%s'}  # formatting x/y-ticks
    fig, ax = plt.subplots()

    heatmap = ax.pcolor(z)
    cbar = plt.colorbar(heatmap, ax=ax)

    step = 1
    xticks = [ticks[xt] % x[i] for i in range(0, len(x), step)]
    yticks = [ticks[yt] % y[i] for i in range(0, len(y), step)]

    ax.set_xticks(np.arange(0, z.shape[1], step) + 0.5, minor=False)
    ax.set_yticks(np.arange(0, z.shape[0], step) + 0.5, minor=False)
    ax.set_xticklabels(xticks, rotation=90, fontsize=10)
    ax.set_yticklabels(yticks, fontsize=10)

    cbar.ax.set_title(zlab)
    ax.set_xlabel(r'%s' % xlab, fontsize=fs)
    ax.set_ylabel(r'%s' % ylab, fontsize=fs)
    ax.set_title(title, fontsize=fs)
    plt.tight_layout()
    if task is None:
        plt.savefig(fig_path+'heatmap_%s.png' % save)
    else:
        plt.savefig(fig_path+'task_%s/heatmap_%s.png' % (task, save))