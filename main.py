"""
 main.py (author: Tina Johnson / git: ??? )
 main processing 
"""
    
import sys 
import matplotlib.pyplot as plt
import time

sys.path.append('preprocessing/')
from audio_conditioning import *
from feature_extraction import *
sys.path.append('models/')
from autoencoder import *
from KNN import *



class SimilarSamples(object):
    
    def __init__(self, 
                 raw_dir,
                 query_name,
                 n_neighbors, 
                 firstTime):
        
        ### USER ###
        self.raw_dir = raw_dir                                  # path to raw audio dataset
        self.query_path = raw_dir+'/temp/audio/'+query_name     # path to query sample
        self.n_neighbors = n_neighbors+1                        # number of samples to return (+1 bc first sample will be itself)
        #self.flip = flip                                       # return least similar instead of most similar
        self.firstTime = firstTime                              # only need to do once: extract feats & train AE
        
        ### INTERNAL ###

        # directories 
        self.processed_dir = raw_dir + '/temp/audio/'         # temp folder to store conditioned audio
        self.dataset_dir = raw_dir + '/temp/'                 # where mel STFT dataset will be sotred
        self.model_dir = raw_dir +'/models/'                  # where AE and KNN models will be stored
        self.encoded_path = self.model_dir+'encoded_set.hdf5' # where the encoded set will be stored
        self.label_path = self.dataset_dir+'labels.txt'
        # audio params
        self.sample_rate = 16000
        self.bit_depth = 16
        self.channels = 1
        self.remove_silence = False
        self.duration = 1.15
        
        # mel stft params
        self.fft_size = 512
        self.pad_len = 1.15
        
        # AE params
        self.train_size = 0.8
        self.input_shape = (128,144,1)
        self.epochs= 50
        self.batch_size = 128
        self.plot=True
        
        # KNN params
        self.metric = "cosine"
        self.algorithm = "brute"
        
        
        None
        
    ############################################################
    ### External functions                                   ###
    ############################################################

    def Process(self):
        
        if (self.firstTime):
            print("CONDITIONING AUDIO...")
            # condition raw audio 
            pre = Preprocess(raw_dir = self.raw_dir, 
                       processed_dir = self.processed_dir,
                       sample_rate = self.sample_rate,
                       bit_depth = self.bit_depth,
                       channels = self.channels,
                       duration = self.duration,
                       remove_silence = self.remove_silence)
            pre.make_samples_list()
            pre.process_audio()
            
            print("waiting...")
            time.sleep(60)
            
            print("EXTRACTING FEATURES...")
            # extract mel stft features, create dataset
            feat = FeatureExtraction(processed_dir = self.processed_dir,
                                    dataset_dir = self.dataset_dir,
                                    fft_size = self.fft_size,
                                    sample_rate = self.sample_rate,
                                    pad_len = self.pad_len,
                                    duration = self.duration)
            feat.computeFeatures()
            feat.saveDataset()
            # delete df after use
            del feat.df_feat ### OPTOMIZE MEM
            
            print("BUILDING AUTOENCODER...")
            # build, train, save AE 
            AE = AutoEncoder(dataset_dir = self.dataset_dir,
                            model_dir = self.model_dir,
                            train_size = self.train_size,
                            input_shape = self.input_shape,
                            epochs = self.epochs,
                            plot = self.plot)

            AE.load_dataset()
            AE.build_model()
            AE.train_model()
            AE.save_encoder()
            
            print("ENCODING AUDIO...")
            # Encode the dataset and query sample
            encoded_set = AE.encoder.predict(AE.x)
            f = h5py.File(self.encoded_path, 'a')
            f.create_dataset('encoded_set', data=encoded_set)
            f.close()
            
            # delete temporary directories 
            os.rmdir(raw_dir+'/temp/') 
        
        ### from hereforth needs to happen every time new query is used ###
        # load array of filenames
        print(self.label_path)
        sample_names = np.loadtxt(self.label_path, dtype='str',delimiter=" ") 
        # grab idx of query sample
        Qidx = np.where(sample_names==self.query_path)
        print(Qidx)
        Qidx = Qidx[0][0]
        print(Qidx)

        # open encoded dataset 
        f = h5py.File(self.encoded_path, 'r')
        encoded_set = f['encoded_set']
        # grab encoded query 
        query = encoded_set[Qidx]
        
        print("PERFORMING KNN CLUSTERING...")
        # KNN 
        KNN = NNeighbors(model_dir = self.model_dir, 
                         n_neighbors = self.n_neighbors,
                         metric = self.metric,
                         algorithm = self.algorithm,
                         encoded_set = encoded_set.value)
        KNN.train_model()

        topk = KNN.return_top_K(Qidx)

        count = 1
        for i in topk:
            print(count, ":", sample_names[i])
            count+=1
       
        # close
        f.close()



def main(folderPath, queryFilename, itemsToReturn, firstTime):
    SimilarSamples(folderPath, queryFilename, itemsToReturn, firstTime).Process()


if __name__ == "__main__":
    try:
        folderPath = str(sys.argv[1])
        queryFilename = str(sys.argv[2])
        itemsToReturn = int(sys.argv[3])
        if (sys.argv[4] == "True" or sys.argv[4] == "true"):
            firstTime = True
        else:
            firstTime = False
        main(folderPath, queryFilename, itemsToReturn, firstTime)
    except:
        print("Program Stopped or Failed")

