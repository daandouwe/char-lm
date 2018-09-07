import os
import codecs
import string


NAMES_DIR = 'names'

CITIES_DIR = 'cities'


# Which is the same as this set.
CHARS = set(string.ascii_lowercase + "~-'`" + ' ' + '0123568' + '()')


PAD_CHAR = '~'


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


LANGUAGES = [
    'Arabic',     'English',    'Irish',      'Polish',
    'Chinese',    'French',     'Italian',    'Portuguese',
    'Czech',      'German',     'Japanese',   'Russian',
    'Dutch',      'Greek',      'Korean',     'Scottish',
    'Spanish',    'Vietnamese'
]


# All characters used in the validation and test sets.
def all_chars(NAME):
    all_lines = \
        [line.strip() for line in open(os.path.join('data', NAME, 'val', f'{NAME}_val.txt')).readlines()] + \
        [line.strip() for line in open(os.path.join('data', NAME, 'test', f'{NAME}_test.txt')).readlines()]
    return set('~'.join(all_lines))


def read_train(path, order):
    with open(path) as f:
        data = f.readlines()
    pad = PAD_CHAR * order
    data = ''.join((pad + line.strip() for line in data))
    return data


def read_test(path):
    with open(path) as f:
        return [line.strip() for line in f.readlines()]


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


if __name__ == '__main__':
    make_utf8(DIR)
    test_open(DIR)
