import os
import codecs

import numpy as np

np.random.seed(42)

COUNTRIES = [
    'af',
    'cn',
    'de',
    'fi',
    'fr',
    'in',
    'ir',
    'pk',
    'za'
]


def make_utf8():
    """The original data throws decoding errors."""
    for set in ('train', 'val'):
        for country in COUNTRIES:
            path = os.path.join('cities', set, f'{country}.txt')
            with codecs.open(path, "r", encoding='utf-8', errors='ignore') as fdata:
                data = fdata.read()
            with open(path, 'w') as f:
                print(data, file=f, end='')
    path = os.path.join('cities', 'test', 'cities_test.txt')
    with codecs.open(path, "r", encoding='utf-8', errors='ignore') as fdata:
        data = fdata.read()
    with open(path, 'w') as f:
        print(data, file=f, end='')


def make_dev():
    pairs = []
    for country in COUNTRIES:
        path = os.path.join('cities', 'val', f'{country}.txt')
        with open(path) as f:
            data = f.readlines()
            pairs.extend((name.strip(), country) for name in data)
    np.random.shuffle(pairs)

    cities, labels = zip(*pairs)

    path = os.path.join('cities', 'val', 'cities_val.txt')
    with open(path, 'w') as f:
        print('\n'.join(cities), file=f)

    path = os.path.join('cities', 'val', 'labels_val.txt')
    with open(path, 'w') as f:
        print('\n'.join(labels), file=f)


def main():
    make_utf8()
    make_dev()

if __name__ == '__main__':
    main()
