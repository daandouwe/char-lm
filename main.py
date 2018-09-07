#!/usr/bin/env python
import argparse
import os
import string
from collections import defaultdict, Counter
import itertools

import numpy as np

from plot import confusion_matrix, plot_grid
from utils import DIR, CHARS, COUNTRIES, PAD_CHAR, read_train, read_test
from model import CharNgram


def train_char_lm(path, order, k, interpolate):
    model = CharNgram(order=order, vocab=CHARS)
    model.train(path, add_k=k)
    if interpolate:
        model.interpolate()
    return model


def train_char_lm_(path, order=3, k=1):
    def normalize(counter):
        s = float(sum(counter.values()))
        return dict((c, cnt/s) for c, cnt in counter.items())

    def add_k(lm, order, k):
        all_histories = list(itertools.product(CHARS, repeat=order))
        all_histories += list(lm.keys())
        all_histories = set(all_histories)
        for history in all_histories:
            history = ''.join(history)
            for char in CHARS:
                lm[history][char] += k
        return lm

    data = read_train(path, order)
    lm = defaultdict(Counter)
    for i in range(len(data)-order):
        history, char = data[i:i+order], data[i+order]
        lm[history][char] += 1
    if k > 0:
        lm = add_k(lm, order, k)
    outlm = dict((hist, normalize(chars)) for hist, chars in lm.items())
    return outlm


def train_models(data_dir, order, k, interpolate):
    models = dict()
    for country in COUNTRIES:
        # print(country)
        path = os.path.join(data_dir, 'train', f'{country}.txt')
        lm = train_char_lm(path, order, k, interpolate)
        models[country] = lm
    return models


def perplexity(data, lm, order):
    logprob = 0
    pad = PAD_CHAR * order
    data = pad + data + pad
    for i in range(len(data)-order):
        history, char = data[i:i+order], data[i+order]
        prob = lm.prob(history, char)
        # prob = lm[history][char]
        logprob += np.log(prob)
    logprob /= (i + 1)
    return np.exp(-logprob)


def score(cities, countries, models, order):
    scores = dict((city, dict()) for city in cities)
    for country in countries:
        for city in cities:
            scores[city][country] = perplexity(city, models[country], order)
    return scores


def predict(cities, scores):
    predicted = []
    for city in cities:
        # Note: Perplexity! Lower is better.
        prediction = min(scores[city], key=(lambda country: scores[city][country]))
        predicted.append(prediction)
    return predicted


def print_predictions(cities, labels, scores, n=10):
    for city, label in zip(cities[:n], labels[:n]):
        print(city, f'({label})')
        for country, val in sorted(scores[city].items(), key=lambda t: t[1]):
            print(f'  {country} {val:.2f}')
        print()


def write_predictions(predicted, fname, folder='pred'):
    path = os.path.join(folder, fname)
    with open(path, 'w') as f:
        print('\n'.join(predicted), file=f)


def evaluate(gold, pred):
    return 100 * sum(p == g for p, g in zip(gold, pred)) / len(pred)


def grid_search(data_dir, cities, gold, orders, ks, random=False):
    accuracies = dict()
    for order in orders:
        if random:
            ks = np.random.uniform(0, 10, 10)
        for k in ks:
            print(f'order {order} k {k:>5.2f}')
            models = train_models(data_dir, order, k)
            scores = score(cities, COUNTRIES, models, order)
            pred = predict(cities, scores)
            accuracy = evaluate(gold, pred)
            accuracies[(order, k)] = accuracy
    return accuracies


def main(args):
    models = train_models(args.data, args.order, args.add_k, args.interpolate)

    cities_path = os.path.join(args.data, 'val', 'cities_val.txt')
    labels_path = os.path.join(args.data, 'val', 'labels_val.txt')
    cities = read_test(cities_path)
    gold = read_test(labels_path)

    scores = score(cities, COUNTRIES, models, args.order)
    pred = predict(cities, scores)

    write_predictions(pred, fname='val_pred.txt')

    print('Some predictions:')
    print_predictions(cities, gold, scores)

    accuracy = evaluate(gold, pred)
    print(f'Validation accuracy: {accuracy:.2f}')

    path = os.path.join('image', f'confusion.n{args.order}k{args.add_k}.png')
    confusion_matrix(gold, pred, out=path)

    if args.grid_search:
        print('Grid search to find best parameters...')
        accuracies = grid_search(
            args.data, cities, gold,
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
    parser.add_argument('--data', type=str, default=DIR)
    parser.add_argument('--interpolate', action='store_true')
    parser.add_argument('--grid-search', action='store_true')
    parser.add_argument('--random-grid', action='store_true')
    parser.add_argument('--plot-grid', action='store_true')
    parser.add_argument('-n', '--order', type=int, default=2)
    parser.add_argument('-k', '--add_k', type=float, default=0)
    args = parser.parse_args()

    main(args)
