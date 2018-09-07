#!/usr/bin/env python
import sys
import os
import string
from collections import defaultdict, Counter
import itertools

import numpy as np

from utils import read_train, read_test


class CharNgram(dict):
    def __init__(self, order=3, vocab={}):
        self.order = order
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.is_unigram = (order == 1)
        self.k = 0
        self._add_k = False
        self._interpolate = False

    def train(self, path, add_k=0):
        def normalize(counter):
            total = float(sum(counter.values()))
            return dict((char, count/total) for char, count in counter.items())

        data = read_train(path, self.order)
        self.data_path = path
        if self.is_unigram:
            lm = Counter(data)
            outlm = normalize(lm)
        else:
            lm = defaultdict(Counter)
            for i in range(len(data)-self.order):
                history, char = data[i:i+self.order], data[i+self.order]
                lm[history][char] += 1
            outlm = ((hist, normalize(chars)) for hist, chars in lm.items())
        super(CharNgram, self).__init__(outlm)
        self.counts = lm
        if not self.vocab:
            self.vocab = set(data)
            self.vocab_size = len(self.vocab)
        if add_k > 0:
            self._add_k = True
            self.k = add_k

    def __smooth_add_k(self, history, char):
        assert self.k > 0, self.k
        if not all(char in self.vocab for char in set(history + char)):
            return 0
        try:
            self.counts[history]
            count = self.counts[history].get(char, 0)
            total = sum(self.counts[history].values())
        except KeyError:
            count = 0
            total = 0
        prob = (self.k + count) / (self.k*self.vocab_size + total)
        return prob

    def interpolate(self):
        self._interpolate = True
        if self.is_unigram:
            self.lower_model = None
        else:
            self.lower_model = CharNgram(self.order - 1)
            self.lower_model.train(self.data_path)
            self.lower_model.interpolate()  # recursion

    def _smooth_interpolate(self, history, char):
        lmbda = self.witten_bell(history)
        if self.is_unigram:
            lmbda = 0.5  # TODO
            higher = self.get(char, 0)
            lower = 1.0 / self.vocab_size
            prob = lmbda * higher + \
                (1 - lmbda) * lower
        else:
            try:
                 higher = self[history][char]
            except KeyError:
                higher = 0
            lower = self.lower_model._smooth_interpolate(history[1:], char)
            prob = lmbda * higher + \
                (1 - lmbda) * lower
        return prob

    def witten_bell(self, history):
        if self.is_unigram:
            unique_follows = self.counts.get(history, 0)
            total = self.counts.get(history, 0)
        else:
            unique_follows = len(self.counts.get(history, []))
            total = sum(self.counts.get(history, dict()).values())
        if unique_follows == 0:
            if total == 0:
                frac = 1  # n/n = 1 as n -> 0
            else:
                frac = 0
        else:
            frac = unique_follows / (unique_follows + total)
        return 1 - frac

    def prob(self, history, char):
        if self._add_k:
            return self._smooth_add_k(history, char)
        if self._interpolate:
            return self._smooth_interpolate(history, char)
        else:
            dist = self.get(history, 0)
            if not dist:
                return 0
            else:
                return dist.get(char, 0)


if __name__ == '__main__':
    from utils import DIR, CHARS, PAD_CHAR
    if len(sys.argv) > 1:
        order = int(sys.argv[1])
    else:
        order = 3

    model = CharNgram(order=order, vocab=CHARS)
    path = os.path.join(DIR, 'train', 'de.txt')
    model.train(path)

    model.interpolate()
    total = 0
    for char in model.vocab:
        prob = model.prob(order*PAD_CHAR, char)
        total += prob
    print(total)
    lmbda = model.witten_bell(order*PAD_CHAR, char)
    # print(lmbda)
