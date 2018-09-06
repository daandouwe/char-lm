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

# Evaluation
The students hand in their test-set predictions. They are evaluated by the accuracy on this set. They can use dev-set for development and grid-search. (Highest score gets bonus?)

# More applications
* Name classification: [Classifying Names with a Character-Level RNN](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) (PyTorch tutorial)
* Authorship attribution: [Language Independent Authorship Attribution using Character Level Language Models](http://www.aclweb.org/anthology/E/E03/E03-1053.pdf)

# TODO
- [ ] Add-k smoothing
- [ ] Interpolation smoothing (backoff, Witten-Bell)
