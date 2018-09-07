#!/usr/bin/env python
import os
import sys

import numpy as np


LANGUAGES = (
    'Arabic',     'English',    'Irish',      'Polish',
    'Chinese',    'French',     'Italian',    'Portuguese',
    'Czech',      'German',     'Japanese',   'Russian',
    'Dutch',      'Greek',      'Korean',     'Scottish',
    'Spanish',    'Vietnamese',
)


def main(dir):
    data = []
    for lang in LANGUAGES:
        path = os.path.join(dir, lang + '.txt')
        with open(path) as f:
            for line in f.readlines():
                data.append((lang, line.strip()))
    np.random.shuffle(data)

    size = len(data) // 20
    val, test, train  = data[:size], data[size:2*size], data[2*size:]

    labels_val, names_val = zip(*val)
    val_path = os.path.join(dir, 'val', '{}_val.txt')

    with open(val_path.format('names'), 'w') as f:
        print('\n'.join(names_val), file=f)
    with open(val_path.format('labels'), 'w') as f:
        print('\n'.join(labels_val), file=f)

    test_path = os.path.join(dir, 'test', '{}_test.txt')
    labels_test, names_test = zip(*test)
    with open(test_path.format('names'), 'w') as f:
        print('\n'.join(names_test), file=f)
    with open(test_path.format('labels'), 'w') as f:
        print('\n'.join(labels_test), file=f)

    train_dict = dict((lang, []) for lang in LANGUAGES)
    for lang, name in train:
        train_dict[lang].append(name)

    for lang in LANGUAGES:
        path = os.path.join(dir, 'train', lang + '.txt')
        with open(path, 'w') as f:
            print('\n'.join(train_dict[lang]), file=f)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        dir = sys.argv[1]
        main(dir)
    else:
        exit('Specify data dir.')
