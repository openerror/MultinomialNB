# MultinomialNB
## Introduction
Clean and well-documented implementation in Python of a naive Bayes classifier, as an [assignment](http://ron.artstein.org/csci544-2019/coding-1.html) for [CSCI 544](http://ron.artstein.org/csci544-2019/) at the University of Southern California.

Goal of assignment: classify between truthful/deceptive and positive/negative hotel reviews
extracted from travel websites. I have chosen to implement a single model to distinguish
between the **2 x 2 = 4** classes.

Training data is contained in directory `op_spam_training_data`, and was extracted from academic
literature by course instructor. CC license allows redistribution and reuse for other purposes;
please refer to READMEs within the data directory for more details.

Code was written assuming Python 3. The only third-party libraries used are [NumPy](http://www.numpy.org/) and a few performance metric functions from [scikit-learn](http://scikit-learn.org/).

## File Description
### `Tokenizer.py`
Custom word-level Python class for turning documents (strings) into integer tokens; capable of
filtering out stop words and punctuation.

### `MultinomialNB.py`
Multinomial naive Bayes classifier, implemented as a Python class.

### `nblearn3.py`
Trains and tests model on development data; measures classification performance using F1 score.

### `nbclassify.py`
Runs classifier trained in `nblearn3.py` on test data; only works on Vocareum grading platform.

### `data_utils.py`
Various helper functions for reading training data, converting it to NumPy arrays etc.
