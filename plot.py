import numpy as np
import matplotlib.pylab as plt
import matplotlib.ticker as ticker


def confusion_matrix(gold, pred, out='confusion.pdf'):

    all_categories = sorted(list(set(gold)))
    n_categories = len(all_categories)
    c2i = dict((c, i) for i, c in enumerate(all_categories))

    confusion = np.zeros((n_categories, n_categories))

    for p, g in zip(gold, pred):
        confusion[c2i[g]][c2i[p]] += 1

    # Normalize by dividing every row by its sum
    confusion = confusion / confusion.sum(1)

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion, vmin=0, vmax=1)
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.savefig(out)


def plot_grid(xs, ys, zs, out='grid.pdf'):
    range = max(zs) - min(zs)
    scale = lambda x: (x - min(zs)) / range

    fig, ax = plt.subplots()
    area = [30 * scale(z) for z in zs]  # 0 to 15 point radii
    alpha = [scale(z) for z in zs]  # 0 to 15 point radii

    alphas = np.linspace(0.1, 1, len(xs))
    rgba_colors = np.zeros((len(xs), 4))
    # for red the first column needs to be one
    rgba_colors[:,0] = 1.0
    # the fourth column needs to be you
    rgba_colors[:, 3] = alphas

    ax.scatter(xs, ys, s=area, color=rgba_colors)
    ax.set_yscale('log')
    plt.savefig(out)
