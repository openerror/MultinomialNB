#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For classifying reviews in test data, and writing results to file
Would not work outside of Vocareum grading platform without some modification
"""

from MultinomialNB import MultinomialNB
from data_utils import list_to_numpy
import sys, dill, glob, os

if __name__ == "__main__":
    # Read test data 
    test_corpus = []
    all_files = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))
    
    for file in all_files:
        with open(file, 'r') as f:
            test_corpus.append(f.read())
    
    # Tokenize test data; must reuse the same Tokenizer used for training data
    with open("tok.pickle", "rb") as g:
        tok = dill.load(g)
    
    tokenized_test_corpus = tok.transform(test_corpus)
    X_test = list_to_numpy(tokenized_test_corpus, tok)
    
    # Load model and make predictions
    nb_clf = MultinomialNB()
    nb_clf.load_model("nbmodel.txt")
    y_pred = nb_clf.predict(X_test)

    # Write predictions to file
    # Format of each line: truthful/deceptive positive/negative path_to_file
    label_to_str = {0: "truthful positive", 1: "deceptive positive",
                    2: "truthful negative", 3: "deceptive negative"}
    
    with open("nboutput.txt", "w") as h:
        for label, path in zip(y_pred, all_files):
            label_str = label_to_str[label]
            output_str = label_str + " " + path + "\n"
            h.write(output_str)