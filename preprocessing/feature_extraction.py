
# coding: utf-8

# In[1]:


"""
 feature_extraction.py (author: Tina Johnson / git: ShimaBanana )
 This class converts processed aduio into STFT for autoencoder input
"""


# In[2]:


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

# np.__version__
# pd.__version__
# h5py.__version__


class FeatureExtraction(object):

    def __init__(self,
                 processed_dir,
                 dataset_dir,
                 fft_size = 512,
                 sample_rate=16000,  
                 pad_len = 1,
                 duration=1.15):
        
        # Directories
        self.processed_dir = processed_dir
        self.dataset_dir = dataset_dir
        
        # audio settings
        self.fft_size = fft_size
        self.sample_rate = sample_rate
        self.pad_len = pad_len
        self.duration = duration
        
        # internal 
        self.df_feat = None
        self.dataset_fname = dataset_dir+'dataset.hdf5'
        self.labels_fname = dataset_dir+'labels.txt'
        
        None
        
    ############################################################
    ### External functions                                   ###
    ############################################################

    def computeFeatures(self):
        
        # create list of sample paths
        samples_list = [] 
        for path, subdirs, files in os.walk(self.processed_dir):
            for filename in files:
                f = os.path.join(path, filename)
                if not "._" in f or not ".DS_Store" in f:
                    if ".wav" in f:
                        samples_list.append(f)
        
        # create DataFrame to store the features
        self.df_feat = pd.DataFrame(columns = [['mel_stft'] + ['sample_path']])
        feat_idx = 0
        
        print("converting audio to mel stft...")
        # take time
        start_time = timeit.default_timer()
        for index in (np.arange(0, len(samples_list))): 
            sample_path = samples_list[index]
            
            try:
                # read pcm data, make sure samplerates match
                y, fs = sf.read(sample_path)
                if (fs!=self.sample_rate):
                    print("ERROR: samplerate mismatch!")
                # pad sample in case below expected duration
                pad_len_samples = int(self.sample_rate * self.pad_len)
                y = np.pad(y, (0,pad_len_samples), 'constant')
                # trim to expected duration
                y = y[:int(self.sample_rate * self.duration)]
                # extract stft and convert to mel
                stft = librosa.core.stft(y, n_fft=self.fft_size)
                mag = np.abs(stft)
                D = mag**2
                mel_stft = librosa.feature.melspectrogram(S=D)
                try:
                    mel_stft = np.reshape(mel_stft, (128, 144, 1))
                except Expection as shapeMismatch:
                    print("ERROR: input shape mismatch!")
                    pass
                
                # save in dataframe
                self.df_feat.at[feat_idx,'mel_stft'] = mel_stft
                self.df_feat.at[feat_idx, 'sample_path'] = sample_path
                
                feat_idx += 1
                #print("Converted file " + str(feat_idx) + " out of " + str(len(samples_list)))
                
            except Exception as inst:
                print(inst)
                pass
            
        elapsed = timeit.default_timer() - start_time
        print('feature extraction time: ', elapsed)
        print(self.df_feat.info())
    
    
    def saveDataset(self):
        # create array of features
        x = np.zeros([len(self.df_feat),128,144,1])
        y = np.empty(len(self.df_feat), dtype='object')
        for i in self.df_feat.index:
            x[i] = self.df_feat['mel_stft'][i]
            y[i] = self.df_feat['sample_path'][i]
            
        # power to dB conversion
        for i in np.arange(0, len(x)):
            x[i, :, :, 0] = librosa.amplitude_to_db(x[i, :, :, 0],ref=np.max)
        #print("input shape:", x.shape)
         
        # save dataset
        print("saving mel stft dataset to", self.dataset_fname)
        f = h5py.File(self.dataset_fname, 'a')
        try:
            f.create_dataset('mel_stft', data=x)
        except:
            print("Dataset already exists.")
        # save labels to txt
        np.savetxt(self.labels_fname, y, delimiter=" ", fmt="%s") 
        f.close()


