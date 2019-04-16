
"""
 train_autoencoder.py (author: Tina Johnson / git: ShimaBanana )
 This class will open a trained autoencoder
 Or build and train a new one
"""


# In[11]:


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

from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Dense, Flatten
from keras.optimizers import Adadelta
from keras.callbacks import Callback, ModelCheckpoint
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage


from numpy.random import randn


class PlotCurrentProgress(Callback):
    def __init__(self, img_dir,test_data):
        self.test_data = test_data
        self.img_dir=img_dir
        self.predhis = []
        self.targets = []

    def on_epoch_end(self, epoch, logs={}):
        x_test = self.test_data
        x_test_pred = self.model.predict(x_test)
        mag_test = x_test_pred[0, :, :, 0]
        librosa.display.specshow(mag_test,y_axis='mel', fmax=8000,x_axis='time', cmap='magma')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel spectrogram')
        plt.tight_layout()
        # save image
        fname = str(self.img_dir+'epoch_'+str(epoch)+'.png')
        plt.savefig(fname)
        plt.show()
        plt.clf()



class AutoEncoder(object):
    
    def __init__(self,
                 dataset_dir,
                 model_dir,
                 train_size = 0.8, # meaning 80/20 test/train split
                 input_shape = (128,144,1),
                 epochs=20,
                 batch_size = 128,
                 plot=True):

        # Directories
        self.dataset_dir = dataset_dir
        self.model_dir = model_dir
        
        # model params
        self.train_size = train_size
        self.input_shape = input_shape
        self.epochs=epochs
        self.batch_size = batch_size
        self.plot=plot
        
        # internal
        self.x = None
        self.Model = None
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.autoencoder_json_path = model_dir+'autoencoder.json'
        self.autoencoder_weights_path = model_dir+'autoencoder.h5'
        self.encoder_json_path = model_dir+'encoder.json'
        self.encoder_weights_path = model_dir+'encoder.h5'
        self.decoder_json_path = model_dir+'decoder.json'
        self.decoder_weights_path = model_dir+'decoder.h5'
        self.checkpoint_dir = model_dir+'/checkpoints/'
        self.img_dir = model_dir+'/imgs/'

    ############################################################
    ### External functions                                   ###
    ############################################################

    def load_dataset(self):
        
        print("Loading dataset...")
        dataset_fname = self.dataset_dir+'/dataset.hdf5'
        f = h5py.File(dataset_fname, 'r+')
        self.x = f['mel_stft']
        
        ## CONVERT POWER TO DB ##
        # power to dB
        #for i in np.arange(0, len(self.x)):
        #    self.x[i, :, :, 0] = librosa.amplitude_to_db(self.x[i, :, :, 0],ref=np.max)
        
        # get input dimensions   
        image_h = self.x.shape[1]
        image_w = self.x.shape[2]
        channels = self.x.shape[3]
        #print("h:", image_h, "w:", image_w, "c:", channels)
        # close file
        f.close
        
#         # PLOT
#         test = x_test[0, :, :, 0]
#         # plt.figure(figsize=(10, 4))
#         librosa.display.specshow(test, y_axis='log', x_axis='time', cmap="magma")
#         plt.title('Power spectrogram')
#         plt.colorbar(format='%+2.0f dB')
#         plt.tight_layout()

        print("Loaded dataset")
    
    def load_autoencoder(self):
        
        print("Loading autoencoder...")
        # load json and create model
        try:
            json_file = open(self.autoencoder_json_path, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.autoencoder = model_from_json(loaded_model_json)
            # load weights into new model
            self.autoencoder.load_weights(self.autoencoder_weights_path)
        except:
            print("ERROR: model not found!")
            pass
        print("Loaded model from disk")
        
    def load_encoder(self):
        
        print("Loading model...")
        # load json and create model
        try:
            json_file = open(self.encoder_json_path, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.autoencoder = model_from_json(loaded_model_json)
            # load weights into new model
            self.autoencoder.load_weights(self.encoder_weights_path)
        except:
            print("ERROR: model not found!")
            pass
        print("Loaded model from disk")
        
    def load_decoder(self):
        
        print("Loading model...")
        # load json and create model
        try:
            json_file = open(self.decoder_json_path, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.autoencoder = model_from_json(loaded_model_json)
            # load weights into new model
            self.autoencoder.load_weights(self.decoder_weights_path)
        except:
            print("ERROR: model not found!")
            pass
        print("Loaded model from disk")
        
    def build_model(self):  
        
        input_img = Input(shape=(128, 144, 1))

        #Build the endoder piece
        encoded = Conv2D(nb_filter=16, nb_row=3, nb_col=3, input_shape=(128, 144, 1), activation='relu', border_mode='same')(input_img)
        encoded = MaxPooling2D((2, 2), border_mode='same')(encoded)
        encoded = Conv2D(nb_filter=8, nb_row=3, nb_col=3, activation='relu', border_mode='same')(encoded)
        encoded = MaxPooling2D((2, 2), border_mode='same')(encoded)
        #encoded = Dropout(0.25)(encoded)
        encoded = Conv2D(nb_filter=8, nb_row=3, nb_col=3, activation='relu', border_mode='same')(encoded)
        encoded = MaxPooling2D((2, 2), border_mode='same')(encoded)
        #encoder = Model(input=input_img, output=encoded)
        #encoded_imgs = encoder.predict(x_train)
        #plt.imshow(encoded_imgs[0].reshape(encoded_imgs[0].shape[0], encoded_imgs[0].shape[1]*encoded_imgs[0].shape[2]).T)

        #Build the decoder piece
        #output_encoder_shape = encoder.layers[-1].output_shape[1:]
        #decoded_input = Input(shape=output_encoder_shape)
        #decoded_output = autoencoder.layers[-1](decoded_input)  # single layer
        
        decoded = Conv2D(nb_filter=8, nb_row=3, nb_col=3, input_shape=(4, 4, 8), activation='relu', border_mode='same')(encoded)
        decoded = UpSampling2D((2, 2))(decoded)
        decoded = Conv2D(nb_filter=8, nb_row=3, nb_col=3, activation='relu', border_mode='same')(decoded)
        decoded = UpSampling2D((2, 2))(decoded)
        #decoded = Dropout(0.5)(decoded)
        decoded = Conv2D(nb_filter=16, nb_row=3, nb_col=3, activation='relu', border_mode='same')(decoded)
        decoded = UpSampling2D((2, 2))(decoded)
        #decoded = Conv2D(nb_filter=1, nb_row=3, nb_col=3, activation='sigmoid', border_mode='same')(decoded)
        decoded = Conv2D(nb_filter=1, nb_row=3, nb_col=3, activation='linear', border_mode='same')(decoded)

        #decoder = Model(decoded_input, decoded)
        # or -- decoder = Model(decoded_input, decoded_output)
        
        #decoded_imgs = decoder.predict(x_train)
        #plt.imshow(decoded_imgs[0].reshape(decoded_imgs[0].shape[0], decoded_imgs[0].shape[1]))

        #Create learning rate schedule and add it to the optimizer of choice
        learning_rate = 1.0
        epochs = 50
        decay_rate = learning_rate / epochs
        #rho and epsilon left at defaults
        adadelta = Adadelta(lr=learning_rate, rho=0.95, epsilon=1e-08, decay=decay_rate)

        #Build the full autoencoder
        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer=adadelta, loss='mse', metrics=['accuracy'])
        self.Model = Model
        self.autoencoder = autoencoder
        self.encoder = Model(self.autoencoder.input, self.autoencoder.layers[6].output)
        
        print("Model built")
        autoencoder.summary()
        
        
    def train_model(self):  

        # train test split
        train_size = int(len(self.x)*self.train_size)
        x_train = self.x[:train_size]
        x_test = self.x[train_size:]
        
        # callbacks (checkpoint and plot)
        if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
        chkfilepath = self.checkpoint_dir+"autoencoder-best_fit.hdf5"
        callbacks = [ModelCheckpoint(chkfilepath, monitor='val_loss', save_best_only=True, verbose=0)]
        if self.plot:
            if not os.path.exists(self.img_dir):
                os.makedirs(self.img_dir)
            test_sample = x_test[:1, :, :, :] # pick the first test sample
            callbacks = [ ModelCheckpoint(chkfilepath, monitor='val_loss', save_best_only=True, verbose=0),
                         PlotCurrentProgress(img_dir=self.img_dir,test_data=test_sample)]
        # train
        self.autoencoder.fit(x_train, x_train, nb_epoch=self.epochs, batch_size=self.batch_size, shuffle=True, 
                             validation_data=(x_test, x_test), callbacks=callbacks)
        
        # copy to self 
        self.encoder = self.Model(self.autoencoder.input, self.autoencoder.layers[6].output)
        
        
    def save_autoencoder(self):
        
        print("Saving autoencoder...")
        # serialize model to JSON
        model_json = self.autoencoder.to_json()
        with open(self.autoencoder_json_path, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.autoencoder.save_weights(self.autoencoder_weights_path)
        print("Saved autoencoder to disk")
        
    def save_encoder(self):
        
        print("Saving encoder...")
        # serialize model to JSON
        model_json = self.encoder.to_json()
        with open(self.encoder_json_path, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.encoder.save_weights(self.encoder_weights_path)
        print("Saved encoder to disk")
        
    def save_decoder(self):
        
        print("Saving decoder...")
        # serialize ENCODER model to JSON
        model_json = self.decoder.to_json()
        with open(self.decoder_json_path, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.decoder.save_weights(self.decoder_weights_path)
        print("Saved decoder to disk")

