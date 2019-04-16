# Sound-Distance-with-Autoencoders-KNN
Given a user specified audio wav file,  this pipeline will search through a user specified database of audio wav files and return the top k most "similar" files to the query file

## The Algorithm
  1. Preprocess audio dataset (convert sample rate, bit depth, sum to mono, trim duration) 
  2. Extract features (convert audio to mel spectrogram) 
  3. Train autoencoder (on entire dataset)
  4. Encode audio dataset
  5. Perform KNN and return top K most similar samples

## Libraries Used

SoX - 1.3.7 (http://sox.sourceforge.net)
numpy - 1.14.5 
pandas - 0.20.3 
soundfile - 0.10.2
h5py - 2.9.0
matplotlib - 2.2.2
librosa - 0.6.2
sklearn - 0.19.1
keras - 2.2.4
  
## To Run

In the command line, navigate to where you downloaded the repository and run this command: 

python main.py [DATASET_FOLDER_PATH] [QUERY_FILENAME] [NUM OF ITEMS TO RETURN] [FIRSTTIME]

Example: python main.py /home/audio/samples/drums 002_kick.wav 5 True

NOTES: 
  - The query file must be in the dataset folder!!! 
  - Steps 1 through 4 (in the algorithm above) only need to happen once per dataset. If you want to run the algorithm on a dataset that has already been encoded but just use a different query, set FIRSTTIME = False 
