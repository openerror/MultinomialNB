#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for training model and measuring its classification performance (F1-score)
"""

from data_utils import generate_training_samples, list_to_numpy, save_model
from MultinomialNB import MultinomialNB
from Tokenizer import Tokenizer
import sys, dill

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

import numpy as np

if __name__ == "__main__":
    # Stop words come from the top few dozens frequent tokens identified by Tokenizer
    # All should be grammatical constructs with little semantic meaning
    stop_words_custom = ['a', 'and', 'the', 'is', 'am', 'are', 'he', 'she', 'it', 'to', 'an']
    
    #priors, training_documents, training_labels = generate_training_samples(sys.argv[1])
    priors, training_documents, training_labels = generate_training_samples("op_spam_training_data/")
    
    # Build Tokenizer and turn training documents into integer tokens
    tok = Tokenizer(num_tokens=None, stop_words=stop_words_custom)
    tokenized_train = tok.fit_transform(training_documents)
    
    # Convert training samples and labels to numpy arrays
    X = list_to_numpy(tokenized_train, tok)
    y = np.asarray(training_labels)
    
    # Split off developmental data
    # Fixed random_state = 42 for DEBUG purposes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                        random_state=49)
    # Fit model on training data
    nb_clf = MultinomialNB()
    nb_clf.fit(X_train, y_train, alpha=0.9)
    
    # Make predictions on developmental data
    y_pred = nb_clf.predict(X_test)
    
    # Print F1 score for each of the four classes, and overall accuracy
    print(f1_score(y_test, y_pred, average=None))
    print(accuracy_score(y_test, y_pred))
    
    # Save model parameters --- priors and conditional probabilities
    save_model("nbmodel.txt", nb_clf, tok)
    
    # Save Tokenizer; will need it for test data
    with open("tok.pickle", "wb") as g:
        dill.dump(tok, g)