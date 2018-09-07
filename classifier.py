#!/usr/bin/env python
import os
from collections import defaultdict, Counter

import numpy as np

from ngram import CharNgram
from utils import PAD_CHAR, all_chars, read_train


class LangModelClassifier:
    def __init__(self, classes, class_dir, order, pad_char=PAD_CHAR):
        self.classes = classes
        self.class_dir = class_dir
        self.order = order
        self.models = dict()
        self.vocab = set()
        self.pad_char = pad_char

    def train_lang_model(self, path, k, interpolate):
        model = CharNgram(order=self.order, vocab=all_chars(self.class_dir))
        model.train(path, add_k=k)
        if interpolate:
            model.interpolate()
        return model

    def train(self, data_dir, k=0, interpolate=False):
        models = dict()
        for clas in self.classes:
            path = os.path.join(data_dir, self.class_dir, 'train', f'{clas}.txt')
            lm = self.train_lang_model(path, k, interpolate)
            models[clas] = lm
            self.vocab.update(lm.vocab)
        self.models = models

    def perplexity(self, data, clas):
        pad = self.pad_char * self.order
        data = pad + data + pad
        lm = self.models[clas]
        logprob = 0
        for i in range(len(data)-self.order):
            history, char = data[i:i+self.order], data[i+self.order]
            prob = lm.prob(history, char)
            logprob += np.log(prob)
        logprob /= (i + 1)
        return np.exp(-logprob)

    def score(self, data):
        scores = dict((item, dict()) for item in data)
        for clas in self.classes:
            for item in data:
                scores[item][clas] = self.perplexity(item, clas)
        return scores

    def predict(self, data, verbose=False, n_examples=10):
        scores = self.score(data)
        predicted = []
        for item in data:
            # Perplexity! Lower is better.
            prediction = min(scores[item], key=(lambda clas: scores[item][clas]))
            predicted.append(prediction)
        if verbose:
            self.print_predictions(data, predicted, scores, n=n_examples)
        return predicted

    def print_predictions(self, data, predicted, scores, n=10):
        for item, label in zip(data[:n], predicted[:n]):
            print(item, f'({label})')
            for clas, val in sorted(scores[item].items(), key=lambda t: t[1]):
                print(f'  {clas:<12} {val:.2f}')
            print()

    def write_predictions(self, predicted, fname, folder='pred'):
        path = os.path.join(folder, fname)
        with open(path, 'w') as f:
            print('\n'.join(predicted), file=f)

    def evaluate(self, gold, pred):
        return 100 * sum(p == g for p, g in zip(gold, pred)) / len(pred)

    def grid_search(self, data_dir, classes, test_data, gold, orders, ks, random=False):
        accuracies = dict()
        rand_min, rand_max = 0, 10
        format_ks = ks if not random else f'uniform({rand_min}, {rand_max})'
        print(f'Grid search for order in {orders} and k in {format_ks}...')
        for order in orders:
            if random:
                ks = np.random.uniform(rand_min, rand_max, 10)
            for k in ks:
                print(f'order {order} k {k:>5.2f}')
                models = self.train(data_dir, order, k)
                pred = self.predict(test_data, verbose=False)
                accuracy = self.evaluate(gold, pred)
                accuracies[(order, k)] = accuracy
        return accuracies
