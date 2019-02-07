#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from collections import defaultdict

class Tokenizer(object):
    def __init__(self, num_tokens=None, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\n', 
                 stop_words=None, lower=True):
        '''
            Simple Tokenizer for generating
            
            Logic of Tokenizer class:
                1) Split document into tokens (_split_and_lower method)
                2) Count number of tokens in the document, after ignoring stop words (fit method)
                3) Retain only the most common num_tokens ones (fit method)
                4) Encode tokens as integers, and return iterable/array containing them
                    Index 1 would be most frequent token, index 2 the second most frequent...so on.
                    Index 0 is reserved for unknown tokens encountered!
                    (transform method)
            
            Args:
                num_tokens: (int) Retain this amount of the most frequent tokens; None for no restriction
                filters: (str) Remove these characters from each document before tokenization
                    default value contains punctuation and the newline symbol
                stop_words: (list[str]) Ignore these words when tokenizing each document
                lower: (boolean) Whether to turn every English letter encountered into lower case
        '''
        
        self.num_tokens = num_tokens
        self.lower = lower

        if filters:
            self.filters = set(filters)
        else:
            self.filters = None
        
        if stop_words:
            self.stop_words = set(stop_words)
        else:
            self.stop_words = None
        
        self.counter_dict = None
        self.token_to_int = None
    
    def _split_and_lower(self, document):
        ''' 
            Helper function for lowering a string's case and splitting it by whitespace.
            Used by self._proprocess()
        '''
        
        if self.lower:
            return document.lower().split()
        else:
            return document.split()
    
    def _preprocess(self, corpus):
        ''' 
            Helper function wrapping up preprocessing tasks common to self.fit and self.transform 
            Args:
                corpus: list containing each document as a string
            Returns:
                processed_documents: a list of map objects (!!!) representing the string tokens
        '''
        
        # First filter out unwanted symbols, such as punctuation
        if self.filters:
            processed_documents = [''.join([char for char in document if char not in self.filters]) 
                                   for document in corpus]
        else:
            processed_documents = corpus
        
        # Split each document into tokens (strings for now); turn to lower-case if requested
        processed_documents = map(self._split_and_lower, processed_documents)
        
        # Remove stop words
        if self.stop_words:
            processed_documents = [[token for token in doc if token not in self.stop_words] 
                                    for doc in processed_documents]
            
        return processed_documents
    
    def fit(self, corpus):
        '''
           Args:
               corpus: list containing each document as a string
           Output:
               Initializes self.counter_dict and self.token_to_int, with respect to corpus.
               These are dictionaries needed for tokenizing an unseen text.
        '''
        
        # Initialize counter_dict, properly this time
        self.counter_dict = defaultdict(lambda: 0)
        
        # Preprocess corpus by filtering and splitting etc
        processed_documents = self._preprocess(corpus)
        
        # Count occurences of each token within the corpus
        for document in processed_documents:
            for token in document:
                self.counter_dict[token] += 1
        
        # Sort the tokens by descending order of occurences; record in list token_counts
        token_counts = [(key,value) for key,value in self.counter_dict.items()]
        token_counts.sort(key = lambda x: x[1], reverse=True)
        
        if self.num_tokens and (self.num_tokens < len(token_counts)):
            token_counts = token_counts[:self.num_tokens]
        
        # Finally, form str-to-int token mapping
        # Remember each item in token_counts is a tuple: (string, # of occurences)
        # Reserve int(0) for unknown tokens
        self.token_to_int = defaultdict(lambda: 0)
        for ix, info in enumerate(token_counts):
            self.token_to_int[info[0]] = (ix+1)
        
    def transform(self, corpus):
        '''
            Transform documents in corpus into integer tokens
            
            Args:
                corpus: list containing each document as a string
            Returns:
                tokenized_documents: 
                    a list of integer tokens; can be mapped back to strings using 
                    inverse of self.token_to_int
        '''
        
        assert self.counter_dict, "ERROR: Tokenizer hasn't been fitted yet. Aborting."
        assert self.token_to_int, "ERROR: Tokenizer hasn't been fitted yet. Aborting."
        
        # Preprocess corpus by filtering and splitting etc
        processed_documents = self._preprocess(corpus)
        
        # Tokenize into integers using fitted dictionaries
        tokenized_document = [[self.token_to_int[str_token] 
                                if str_token in self.token_to_int else 0
                               for str_token in doc ]
                              for doc in processed_documents]

        return tokenized_document
    
    def fit_transform(self, corpus):
        '''
            Wrapper for achieving fit and transform in one line
        '''
        self.fit(corpus)
        return self.transform(corpus)