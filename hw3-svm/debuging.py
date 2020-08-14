# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 18:50:11 2020

@author: User
"""

import pickle
from collections import Counter
import util

def convert_to_sparse_BOW(review_list):
    bow_of_reviews = []
    for reviews in review_list:
        bow_of_reviews.append(Counter(reviews))
    return bow_of_reviews

class svm_pegasos():
    def __init__(self, lambda_reg = 1):
        if lambda_reg < 0:
            raise ValueError('Regularization penalty should be at least 0.')
        self.lambda_reg = lambda_reg
    
    def fit(self, X):
        self.w_ = dict()
        t = 0
        
        for j in range(len(X)):
            t += 1
            step_size = 1/(t*self.lambda_reg)
            
            w_dot_x = util.dotProduct(self.w_,X[j])
            y = 1 if 1 in X[j] else -1
            
                      
            if y*w_dot_x < 1:
                util.increment(self.w_, (1 - 1/t), self.w_)
                util.increment(self.w_, step_size*y, X[j])
            
            else:
                util.increment(self.w_, (1 - 1/t), self.w_)
                
        return self.w_

data = pickle.load(open('data_processed.p', 'rb'))

bow = convert_to_sparse_BOW(data)

svm = svm_pegasos(1)
svm.fit(bow)