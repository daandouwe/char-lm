# Character ngram language model
A simple character ngram language model to illustrate:
  * MLE probability estimates
  * Smoothing (add-k and interpolation)
  * Perplexity
  * Text production
  * Text classification with perplexity

The code in this repository is based on the homework assignment [Character-based Language Models](http://computational-linguistics-class.org/assignment5.html) and [The unreasonable effectiveness of Character-level Language Models](https://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139).

The classification task and data is taken from [Character-based Language Models](http://computational-linguistics-class.org/assignment5.html), and the data to produce text samples is taken from [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/).

# Setup
To get the required data for this code, run:
```bash
cd data
./get-data.sh
```

# Usage
To run a classification demo on the `cities` dataset, type:
```bash
./main.py classify --interpolate
```
For the `names` dataset, type:
```bash
./main.py classify --dataset names --interpolate
```
Choose the order and add-k smoothing with
```bash
--order 3 --add-k 1
```
To additionally perform grid-search for smoothing parameters, add:
```bash
--grid-search
```

# Excercises
Students implement:
  * text-sampling
  * perplexity computation
  * add-k smoothing
  * interpolation smoothing (and Witten-Bell lambda rule for interpolation)
  * text-classification (training one lm per language)
  * grid-search on dev-set for smoothing parameters

# Text production
```
Under construction
```

# City classification
The character language model can be used to classify text. Have a look at the [cities](https://github.com/daandouwe/char-lm/tree/master/data/cities/train) dataset. For each country in the training dataset (`af`, `cn`, `de`, `fi`, `fr`, `in`, `ir`, `pk`, `za`) we train a char-lm (with smoothing) on the list of given cities. During prediction, we choose the country with the lowest perplexity.

The data (and idea) is taken from the homework assignment [Character-based Language Models](http://computational-linguistics-class.org/assignment5.html).

Here's an example of scores on the dev-set (true country is listed between brackets):
```
Some predictions:
harvanmaki (fi)
  fi 9.44
  ir 14.85
  in 16.09
  af 17.02
  pk 17.29
  za 17.72
  de 19.81
  cn 22.74
  fr 32.47

ditodai dano (pk)
  in 13.65
  za 14.65
  pk 14.77
  de 16.04
  ir 16.37
  cn 16.41
  af 16.76
  fi 19.03
  fr 22.78

shanjiatun (cn)
  cn 6.19
  pk 10.30
  af 10.41
  ir 10.45
  in 11.25
  za 14.07
  de 16.11
  fi 17.01
  fr 25.17

Validation accuracy: 67.32
```
An order 3 model with add-1 smoothing can achieve an accuracy of over 68% (see [grid-search.txt](https://github.com/daandouwe/char-lm/blob/master/grid-search.txt)).

We can also plot a confusion matrix from the predictions:
![confusion](https://github.com/daandouwe/char-lm/blob/master/image/confusion.cities.interpolate.png)

# Name classification
Have a look at the [names](https://github.com/daandouwe/char-lm/tree/master/data/names/train) dataset. For each language in the training dataset (18 in total) we train a char-lm (with smoothing) on the list of given cities. During prediction, we choose the country with the lowest perplexity.

The data is taken from the PyTorch tutorial [Classifying Names with a Character-Level RNN](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) (PyTorch tutorial).

Here's an example of scores on the dev-set (true language is listed between brackets):
```
Some predictions:
Agadjanov (Russian)
  Russian      3.68
  Italian      28.28
  Spanish      33.39
  Portuguese   34.00
  Greek        34.02
  Czech        34.33
  Scottish     43.98
  Vietnamese   46.74
  Polish       47.47
  Chinese      50.61
  Japanese     54.49
  Dutch        54.60
  French       54.83
  English      59.01
  Irish        59.71
  Korean       59.89
  German       83.16
  Arabic       110.12

O'Reilly (Irish)
  Irish        5.81
  English      13.53
  French       20.33
  Dutch        26.08
  Scottish     27.24
  Czech        30.44
  Spanish      33.92
  Polish       37.71
  German       38.89
  Italian      49.22
  Russian      55.22
  Korean       59.47
  Chinese      59.78
  Vietnamese   64.41
  Greek        66.03
  Portuguese   72.55
  Japanese     74.86
  Arabic       83.92

Evelson (English)
  English      4.89
  Russian      7.32
  German       9.89
  Scottish     16.60
  Dutch        17.87
  French       22.83
  Japanese     28.35
  Italian      29.48
  Irish        29.62
  Spanish      33.69
  Korean       35.72
  Arabic       37.60
  Czech        39.45
  Polish       39.60
  Greek        39.92
  Chinese      41.92
  Vietnamese   43.85
  Portuguese   53.67

Issa (Arabic)
  Arabic       2.83
  Japanese     7.49
  English      11.44
  Italian      16.90
  Spanish      17.91
  Greek        18.13
  Russian      22.07
  Portuguese   25.35
  Czech        26.09
  Polish       30.36
  Dutch        32.88
  Irish        37.68
  Vietnamese   42.32
  Chinese      43.05
  Korean       46.34
  German       50.52
  Scottish     53.81
  French       74.46

Sauveterre (French)
  French       5.07
  English      11.35
  Portuguese   13.22
  Spanish      16.34
  Irish        17.10
  Italian      17.14
  German       18.67
  Russian      21.33
  Scottish     22.26
  Dutch        23.31
  Czech        23.50
  Polish       28.41
  Greek        28.92
  Japanese     35.30
  Korean       35.32
  Vietnamese   35.63
  Chinese      36.02
  Arabic       61.76

Subertova (Czech)
  Czech        7.36
  Italian      9.61
  English      14.48
  Spanish      14.67
  Russian      16.68
  Portuguese   19.33
  German       20.61
  Scottish     24.16
  French       24.92
  Polish       24.92
  Irish        25.30
  Dutch        26.42
  Korean       29.28
  Chinese      30.07
  Japanese     31.85
  Greek        36.31
  Vietnamese   39.63
  Arabic       48.81

Validation accuracy: 81.75
```
We can also plot a confusion matrix from the predictions:
![confusion](https://github.com/daandouwe/char-lm/blob/master/image/confusion.names.interpolate.png)

# Evaluation
The students hand in their test-set predictions. They are evaluated by the accuracy on this set. They can use dev-set for development and grid-search. (Highest score gets bonus?)

# More applications
* Authorship attribution: [Language Independent Authorship Attribution using Character Level Language Models](http://www.aclweb.org/anthology/E/E03/E03-1053.pdf)

# TODO
- [X] Add-k smoothing
- [X] Interpolation smoothing (backoff, Witten-Bell)
- [ ] Train lm on shakespeare and linux
- [ ] Sample text
