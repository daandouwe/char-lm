import numpy as np
import matplotlib.pylab as plt
import matplotlib.ticker as ticker


def confusion_matrix(gold, pred, out='confusion.pdf'):
    assert len(gold) == len(pred), f'inconsistent lengths: gold {len(gold)}  pred {len(pred)}'
    assert set(pred) <= set(gold)

    all_categories = sorted(list(set(gold)))

    n_categories = len(all_categories)
    c2i = dict((c, i) for i, c in enumerate(all_categories))

    confusion = np.zeros((n_categories, n_categories))

    for p, g in zip(gold, pred):
        confusion[c2i[g]][c2i[p]] += 1

    # Normalize by dividing every row by its sum
    confusion += 1  # avoid divison by zero
    confusion = confusion / confusion.sum(axis=1)[:, np.newaxis]

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # cax = ax.matshow(confusion, vmin=0, vmax=1)
    cax = ax.matshow(confusion)
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.savefig(out)


def plot_grid(xs, ys, zs, out='grid.pdf', random=False):
    assert len(xs) == len(ys) == len(zs), f'inconsistent lengths: xs {len(xs)}  ys {len(ys)}  zs {len(zs)}'

    range = max(zs) - min(zs)
    scale = lambda x: (x - min(zs)) / range

    fig, ax = plt.subplots()
    area = [60 * np.exp(3*scale(z)) for z in zs]
    alphas = [scale(z) for z in zs]

    rgba_colors = np.zeros((len(xs), 4))  # 4 channels, one alpha
    rgba_colors[:,0] = 1.0  # color red
    rgba_colors[:, 3] = alphas

    ax.scatter(xs, ys, s=area, color=rgba_colors)
    if not random:
        ax.set_yscale('log')
    plt.savefig(out)
