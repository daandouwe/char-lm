# !/usr/bin/env bash

mkdir cities
mkdir cities/test

# 1. Prepare cities dataset.
wget http://computational-linguistics-class.org/downloads/hw5/cities_train.zip
wget http://computational-linguistics-class.org/downloads/hw5/cities_val.zip
wget http://computational-linguistics-class.org/downloads/hw5/cities_test.txt
unzip -q cities_train.zip
unzip -q cities_val.zip
rm cities_train.zip cities_val.zip
mv train cities
mv val cities
mv cities_test.txt cities/test

# Fix some none utf-8 encodings.
./make-cities-utf8.py cities

# Make a dev-set like the test-set.
python process.py
mkdir cities/val/countries
mv cities/val/{af,cn,de,fi,fr,in,ir,pk,za}.txt cities/val/countries


# 2. Prepare names dataset.
mkdir names
wget https://download.pytorch.org/tutorial/data.zip
unzip -q data.zip
mv data/names/* names
rm data.zip
rm -r data

# Make train/dev/test splits.
mkdir names/train names/val names/test
./make-name-data.py names
rm names/*.txt


# 3. Get karpathy's shakespeare and linux datasets.
# mkdir shakespeare linux
# wget http://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt
# wget http://cs.stanford.edu/people/karpathy/char-rnn/linux_input.txt
# mv shakespeare_input.txt shakespeare
# mv linux_input.txt linux
