# Character ngram language model
A simple character ngram language model to illustrate:
  * MLE probability estimates
  * Smoothing (add-k and interpolation)
  * Perplexity
  * Text production
  * Text classification

The code in this repository is based on the homework assignment [Character-based Language Models](http://computational-linguistics-class.org/assignment5.html) and [The unreasonable effectiveness of Character-level Language Models](https://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139).

The classification task and data is taken from [Character-based Language Models](http://computational-linguistics-class.org/assignment5.html), and the data to produce text samples is taken from [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/).

# Setup
To get the required data for this code, run
```bash
./get-data.sh
```
from the project directory.

# Usage
To run a demo, type:
```bash
./main.py
```
To additionally perform grid-search for the best parameters, type:
```bash
./main.py --grid-search
```

# Excercises
Implement:
  * text-sampling
  * perplexity
  * add-k smoothing
  * interpolation smoothing (and Witten-Bell)
  * text-classification (trainging one lm per language)
  * grid-search on dev-set for smoothing parameters

# Text classification
The character language model can be used to classify text. Have a look at the `cities` dataset. For each country in the training dataset (`af`, `cn`, `de`, `fi`, `fr`, `in`, `ir`, `pk`, `za`) we train a char-lm (with smoothing) on the list of given cities. During prediction, we choose the country with the lowest perplexity.

Here's an example of scores on the dev-set (true country is listed between brackets):
```
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
```
An order 3 model with add-1 smoothing can achieve an accuracy of over 68% (see [grid-search.txt](https://github.com/daandouwe/char-lm/blob/master/grid-search.txt)).

We can also plot a confusion matrix from the predictions:
![confusion](https://github.com/daandouwe/char-lm/blob/master/image/confusion.n2k1.0.png)

# Evaluation
The students hand in their test-set predictions. They are evaluated by the accuracy on this set. They can use dev-set for development and grid-search. (Highest score gets bonus?)

# More applications
* Name classification: [Classifying Names with a Character-Level RNN](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) (PyTorch tutorial)
* Authorship attribution: [Language Independent Authorship Attribution using Character Level Language Models](http://www.aclweb.org/anthology/E/E03/E03-1053.pdf)

# TODO
- [ ] Add-k smoothing
- [ ] Interpolation smoothing (backoff, Witten-Bell)
