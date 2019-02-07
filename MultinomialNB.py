#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class MultinomialNB(object):
    '''
        Implements multinomial naive Bayes classifier; useful for discriminating between 
        different types of texts, among other things.
        
        API constructed to be similar to that of scikit-learn's, in order to ease
        implementation and code reuse.
        
        Args:
            None at instance initialization; please initialize model parameters via fit() method
            
    '''
    
    def __init__(self):
        # 1D iterable containing prior probabilities of each class
        self.priors = None
        
        # 2D Numpy array to be fitted
        # Rows: tokens, column: features e.g. integer counts of each word token
        self.params = None
        
        # Return these values when making prediction
        self.unique_labels = None
    
    def fit(self, X, y, alpha=1.0):
        '''
            Fits multinomial naive Bayes model; add-alpha smoothing applied to handle rare words.
            
            Args:
                X: numpy 2D array recording counts of each feature (col) in each document (row)
                y: 1D numpy array or list recording integer class label
                alpha: smoothing parameter; default to 1.0 for simple add-1 smoothing
            Output:
                Fit self.params and self.priors according to X and y
        '''
        
        assert ((alpha <= 1.0) and (alpha > 0.0)), "ERROR: smoothing parameter alpha should have value within [0.0, 1.0]!"
        
        self.unique_labels = np.unique(y)
        
        # Remember: X.shape[1] is the number of unique tokens
        self.params = np.zeros(shape = (X.shape[1], len(self.unique_labels)))
        self.priors = np.zeros(shape = (len(self.unique_labels),))
        
        for ix,label in enumerate(self.unique_labels):
            # Boolean mask for extracting training samples corresponding to label
            mask = (y == label)
            
            # Add-1 smoothing; verified numerically that probabilities column-sum to 1
            token_counts_in_label = (np.sum(X[mask, :], axis=0) + alpha)
            total_tokens_in_label = np.sum(X[mask, :]) + X.shape[1] * alpha
            self.params[:, ix] = token_counts_in_label / total_tokens_in_label
            self.priors[ix] = np.sum(mask)/len(y)
    
    def predict_log_likelihood(self, X):
        '''
            Computes log likelihood for each document, given each class
            
            Args:
                X: numpy 2D array recording counts of each feature (col) in each document (row)
            Returns:
                log_likelihoods: 2D numpy array containing logarthimic likelihoods 
        '''

        log_params = np.log(self.params)
        log_likelihoods = np.dot(X, log_params)
        return log_likelihoods
            
    def predict(self, X):
        '''
            Predicts class labels based on log likelihood
            
            Args:
                X: numpy 2D array recording counts of each feature (col) in each document (row)
            Returns:
                pred_y: 1D numpy array containing predicted class label for each document
        '''
        
        log_likelihoods = self.predict_log_likelihood(X)
        index_to_label = np.argmax(log_likelihoods, axis=1)
        pred_y = np.asarray([self.unique_labels[index] for index in index_to_label])
        
        return pred_y
        
    def load_model(self, load_from_path):
        '''
            Load model info from file. See data_utils for save_model();
            that is not part of MultinomialNB, because it also requires Tokenizer
            object to work!
        '''
        
        with open(load_from_path, "r") as f:
            # First line are unique labels
            self.unique_labels = np.array([int(label) for label in f.readline().split()])
            
            # Second line are prior probabilities
            self.priors = np.array([float(p) for p in f.readline().split()])
            
        # Rest for token conditional probs. per class
        param_cols = [i for i in range(1, len(self.priors)+1)]
        self.params = np.loadtxt(load_from_path, skiprows=2, usecols=param_cols)