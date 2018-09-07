#!/usr/bin/env python

import sys
import os
import codecs


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


def main(dir):
    """The original data throws decoding errors."""
    for set in ('train', 'val'):
        for country in COUNTRIES:
            path = os.path.join(dir, set, f'{country}.txt')
            with codecs.open(path, "r", encoding='utf-8', errors='ignore') as fdata:
                data = fdata.read()
            with open(path, 'w') as f:
                print(data, file=f, end='')
    set = 'test'
    path = os.path.join(dir, set, 'cities_test.txt')
    with codecs.open(path, "r", encoding='utf-8', errors='ignore') as fdata:
        data = fdata.read()
    with open(path, 'w') as f:
        print(data, file=f, end='')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        dir = sys.argv[1]
        main(dir)
    else:
        exit('Specify data dir.')
