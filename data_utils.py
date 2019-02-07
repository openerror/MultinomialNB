#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
    Functions for 
    - preparing text (strings) for input into MultinomialNB classifer
    - 
'''

import os
import os.path as path
import numpy as np
from collections import Counter

def save_model(path, classifier_obj, tokenizer):
    '''
        Save model info to path. Information includes:
            1. unique labels
            2. prior probabilities corresponding to each unique label
            3. token in string form, followed by their conditional probabilities to be in each class
    '''
    
    unique_labels = classifier_obj.unique_labels
    params = classifier_obj.params
    priors = classifier_obj.priors
    str_to_ix = tokenizer.token_to_int
    
    # Will be useful in associating conditional probabilities explicitly with a str
    ix_to_str = {ix:token for token,ix in str_to_ix.items()}
    
    with open(path, "w") as f:
        # Write unique labels
        label_str = ' '.join([str(label) for label in unique_labels]) + '\n'
        f.write(label_str)
        
        # Write prior information
        prior_str = ' '.join([str(p) for p in priors]) + '\n'
        f.write(prior_str)
        
        # Write token conditional probabilities
        # Start by index 0, which corresponds to unknown tokens and is not 
        # explicitly encoded by Tokenizer
        f.write("UNK " + ' '.join([str(val) for val in params[0,:]]) + '\n')
        
        # Write the rest of the tokens
        for ix in range(1, params.shape[0]):
            token = ix_to_str[ix]
            param_str = token + ' ' +  ' '.join([str(val) for val in params[ix,:]]) + "\n"
            f.write(param_str)

def list_to_numpy(tokenized_documents, tokenizer):
    '''
        Convert tokenized corpus into NumPy array, for use in multinomial naive Bayes.
        
        Args:
            tokenized_documents: list containing documents already tokenized into integers
            tokenizer: the Tokenizer object responsible for converting docs to ints
        Returns:
            numpy_documents: NumPy array where each row describes a document, and each column
            corresponds to a unique token. The array element itself records the number of occurences
            of a particular token in a given document
    '''
    
    # Add 1 to account for unknown token (int(0)), which is not recorded in tokenizer
    num_unique_tokens = len(tokenizer.token_to_int)+1
    num_docs = len(tokenized_documents)
    
    # Use 32-bit integer; saves space and should be sufficient anyways
    numpy_documents = np.zeros(shape = (num_docs, num_unique_tokens), dtype=np.int32)
    
    # Count number of tokens in each document, and write to numpy_documents
    token_count_per_doc = [Counter(doc) for doc in tokenized_documents]
    for doc_ix,counter_dict in enumerate(token_count_per_doc):
        for token,count in counter_dict.items():
            numpy_documents[doc_ix, token] = count
    
    return numpy_documents
    

def per_class_helper(class_directory, documents_array, label_array, class_label):
    '''
        Encodes repetitive code for reading actual files belonging to each class.
        Used by generate_training_samples()
        
        Returns:
            - class_count, the number of class instances; helps with calculating priors
    '''
    
    class_count = 0 # Records the number of class instances
    
    for fold in os.listdir(class_directory):
        # If-statement for ignoring special OS files, e.g. .DS_Store
        if fold[0] != '.':
            fold_directory = path.join( class_directory, fold )
        
        for review in os.listdir(fold_directory):
            # If-statement for ignoring special OS files, e.g. .DS_Store
            if review[0] != '.':
                review_path = path.join(fold_directory, review)
                
                # Read entire file as list; each line becomes a string in the list
                with open(review_path, 'r') as f:
                    documents_array.append( f.read() )
                    label_array.append(class_label)
                class_count += 1
    
    return class_count

def generate_training_samples(input_dir):
    '''
        Function hardcoded to generate training data for Naives Bayes classifier
        from specifically the Blackboard archive.
        
        Labels: 0 1 2 3, corresponding to
            - positive and truthful
            - positive and deceptive
            - negative and truthful
            - negative and deceptive
        
        Args:
            - Directory containing training data
        
        Returns:
            - List containing training data: tuples (tokenized review, label)
            - List containing prior probabilities, in the same order as the label
    '''
    
    training_documents = []
    training_labels = []
    
    ## Label 0
    c1 = per_class_helper(path.join(input_dir, "positive_polarity", "truthful_from_TripAdvisor"), 
                          training_documents, training_labels, 0)
    ## Label 1
    c2 = per_class_helper(path.join(input_dir, "positive_polarity", "deceptive_from_MTurk"), 
                          training_documents, training_labels, 1)
    ## Label 2
    c3 = per_class_helper(path.join(input_dir, "negative_polarity", "truthful_from_Web"), 
                          training_documents, training_labels, 2)
    ## Label 3
    c4 = per_class_helper(path.join(input_dir, "negative_polarity", "deceptive_from_MTurk"), 
                          training_documents, training_labels, 3)
    
    total_number_of_docs = (c1 + c2 + c3 + c4)
    priors = list(map(lambda x: x / total_number_of_docs, [c1, c2, c3, c4]))
    
    return priors, training_documents, training_labels