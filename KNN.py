
# coding: utf-8

# In[1]:


"""
 train_autoencoder.py (author: Tina Johnson / git: ??? )
 This class will open a trained autoencoder
 Or it build and train a new one
"""


# In[ ]:


import h5py
import os
import sys
import timeit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import soundfile as sf
import librosa
import librosa.display as libdisp
import IPython.display
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
import pickle
import cmath

from sklearn.neighbors import NearestNeighbors


# In[4]:


class NNeighbors(object):
    
    def __init__(self,
                 model_dir,
                 n_neighbors = 5,
                 metric = "cosine",   # kNN metric (cosine only compatible with brute force)
                 algorithm = "brute", # search algorithm
                 recommendation_method = 2, # 1 = centroid kNN, 2 = all points kNN
                 encoded_set = None):
        
        # directories
        self.model_dir = model_dir
        
        # model params
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.algorithm = algorithm
        self.recommendation_method = recommendation_method
        
        # data
        self.encoded_set = encoded_set
        
        # internal 
        self.knn = None
        
        
        None
    ############################################################
    ### External functions                                   ###
    ############################################################
    
    def train_model(self):
        
        print("Training KNN...")
        # KNN expcets 1D input, must flatten encoded set
        self.encoded_set = self.encoded_set.reshape(len(self.encoded_set), self.encoded_set.shape[1]*self.encoded_set.shape[2]*self.encoded_set.shape[3])
        self.knn = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm=self.algorithm, metric=self.metric)
        self.knn.fit(self.encoded_set)
        print("KNN trained")
    
    def return_top_K(self, Qidx):
        query = self.encoded_set[Qidx]
        query = query.reshape(1,len(query))
        topk = self.knn.kneighbors(query, return_distance=True)
        # since query sample is in dataset, first sample returned will be itself. Remove.
        topk = topk[1][0][1:]
        
        return topk
        
