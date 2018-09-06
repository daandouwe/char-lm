import os
import codecs
import string


DIR = 'data/cities'


# All characters used in the validation and test sets.
DATA_CHARS = set(
    [line.strip() for line in open(os.path.join(DIR, 'val', 'cities_val.txt')).readlines()] + \
    [line.strip() for line in open(os.path.join(DIR, 'test', 'cities_test.txt')).readlines()]
)

# Which is the same as this set.
CHARS = set(string.ascii_lowercase + "~-'`" + ' ' + '0123568' + '()')


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


def test_open(dir):
    for set in ('train', 'val'):
        for country in COUNTRIES:
            path = os.path.join(dir, set, f'{country}.txt')
            with open(path) as f:
                data = f.read()
            print(data)
    path = os.path.join(dir, 'test', 'cities_test.txt')
    with open(path) as f:
        data = f.read()
    print(data)


def make_utf8(dir):
    """The original data throws decoding errors."""
    for set in ('train', 'val'):
        for country in COUNTRIES:
            path = os.path.join(dir, set, f'{country}.txt')
            with codecs.open(path, "r", encoding='utf-8', errors='ignore') as fdata:
                data = fdata.read()
            with open(path, 'w') as f:
                print(data, file=f, end='')
    path = os.path.join(dir, 'test', 'cities_test.txt')
    with codecs.open(path, "r", encoding='utf-8', errors='ignore') as fdata:
        data = fdata.read()
    with open(path, 'w') as f:
        print(data, file=f, end='')


if __name__ == '__main__':
    make_utf8(DIR)
    test_open(DIR)
