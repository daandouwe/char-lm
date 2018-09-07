#!/usr/bin/env python
import argparse
import os
from collections import Counter
import itertools

import numpy as np

from classifier import LangModelClassifier
from utils import COUNTRIES, LANGUAGES, CITIES_DIR, NAMES_DIR, read_test
from plot import confusion_matrix, plot_grid


def main(args):
    if args.dataset == 'cities':
        class_dir = CITIES_DIR
        classes = COUNTRIES
    if args.dataset == 'names':
        class_dir = NAMES_DIR
        classes = LANGUAGES

    data_path = os.path.join(args.data, class_dir, 'val', f'{args.dataset}_val.txt')
    labels_path = os.path.join(args.data, class_dir, 'val', 'labels_val.txt')
    val_data = read_test(data_path)
    val_labels = read_test(labels_path)

    model = LangModelClassifier(classes, class_dir, args.order)
    model.train(args.data, args.add_k, args.interpolate)
    print(f'Made language model classifier for dataset `{args.dataset}`...')
    print(f'Classes: {classes}.')
    print(f'Num classes: {len(classes)}.')
    print()

    print('Some predictions:')
    val_pred = model.predict(val_data, verbose=True, n_examples=args.n_examples)
    model.write_predictions(val_pred, fname='val_pred.txt')

    accuracy = model.evaluate(val_labels, val_pred)
    print(f'Validation accuracy: {accuracy:.2f}')

    name_base = f'confusion.{args.dataset}'
    if args.interpolate:
        name = name_base + '.interpolate'
    else:
        name = name_base + f'.n{args.order}k{args.add_k}'
    ext = '.png'
    name += ext
    path = os.path.join('image', name)
    confusion_matrix(val_labels, val_pred, out=path)

    if args.grid_search:
        print('Grid search to find best parameters...')
        accuracies = model.grid_search(
            args.data,
            classes,
            data,
            labels,
            orders=(1, 2, 3, 4, 5, 6),
            ks=(0.01, 0.1, 1, 10),
            random=args.random_grid
        )
        accuracies = Counter(accuracies).most_common()
        print('Results:')
        for (order, k), acc in accuracies:
            print(f'order {order}  k {k:>5.2f}  acc {acc:.2f}')

        with open('grid-search.txt', 'w') as f:
            for (order, k), acc in accuracies:
                print(f'order {order}  k {k:>5.2f}  acc {acc:.2f}', file=f)

    if args.plot_grid:
        path = os.path.join('image', 'grid.pdf')
        xy, zs = zip(*accuracies)
        xs, ys = zip(*xy)
        plot_grid(xs, ys, zs, out=path, random=args.random_grid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['cities', 'names'])
    parser.add_argument('--data', type=str, default='data')
    parser.add_argument('--interpolate', action='store_true')
    parser.add_argument('--grid-search', action='store_true')
    parser.add_argument('--random-grid', action='store_true')
    parser.add_argument('--plot-grid', action='store_true')
    parser.add_argument('-n', '--order', type=int, default=2)
    parser.add_argument('-k', '--add_k', type=float, default=0)
    parser.add_argument('-e', '--n-examples', type=int, default=10)
    args = parser.parse_args()

    main(args)
