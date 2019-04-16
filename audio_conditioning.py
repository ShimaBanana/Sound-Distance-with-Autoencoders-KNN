"""
 preprocessing.py (author: Tina Johnson / git: ??? )
 This class pre processes aduio duration, sample rate, bit depth, and channels for STFT conversion
"""



import os
import sys
import subprocess
import timeit



class Preprocess(object):

    def __init__(self,
                 raw_dir,
                 processed_dir,
                 sample_rate=16000, 
                 bit_depth=16, 
                 channels=1, 
                 duration=1.15,
                 remove_silence=False):
        
        # Directories
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir 
        
        # audio settings
        self.sample_rate = sample_rate
        self.bit_depth = bit_depth
        self.channels = channels
        self.remove_silence = remove_silence
        self.duration = duration
        
        # internal
        self.samples_list = None
        
        None
        
        
    ############################################################
    ### External functions                                   ###
    ############################################################

    def make_samples_list(self):
        self.samples_list = [] 
        for path, subdirs, files in os.walk(self.raw_dir):
            for filename in files:
                f = os.path.join(path, filename)
                if not "._" in f or not ".DS_Store" in f:
                    if ".wav" in f:
                        self.samples_list.append(f)
        
        #print("processing", len(self.samples_list), "samples")

    def process_audio(self, processed_dir=None, sample_rate=None, 
                      bit_depth=None, channels=None, remove_silence=None, duration=None):
        
        print("converting sample rate, bit depth and summing to mono...")
        print("removing silence and trimming duration...")
        
        cmd_SR = ""
        if self.remove_silence:
            cmd_SR = "silence 1 0.1 .1% -1 0.1 .1"
        
        #cmd = 'sox sample_in -r '+ str(self.sample_rate)+ ' -b '+str(self.bit_depth)+' -c '+str(self.channels)+' sample_out '+ str(cmd_SR)+ ' pad '+str(self.pad_len)+' trim '+str(self.duration)
        cmd = 'sox sample_in -r '+ str(self.sample_rate)+ ' -b '+str(self.bit_depth)+' -c '+str(self.channels)+' sample_out'+ ' trim 0 ' + str(self.duration)
        print(cmd)
        augmentation = ['WAV', cmd]
        count = 0
        start_time = timeit.default_timer()
        for sample in self.samples_list:
            count += 1
            # fet file path
            file_path = os.path.dirname(os.path.realpath(sample))

            # create augmented directory if it doesn't exists
            if not os.path.exists(self.processed_dir):
                os.makedirs(self.processed_dir)

            tmp_cmd = augmentation[1]
            tmp_cmd = tmp_cmd.split()
            tmp_cmd[1] = sample  # full path to input sample
            tmp_cmd[8] = self.processed_dir + '/' + os.path.split(
                sample)[1].split('.')[0] + '.wav'  # full path to output sample
            # run sox command
            # print(tmp_cmd)
            subprocess.Popen(tmp_cmd)
            #print("processed", count, "out of", len(self.samples_list))
            
        elapsed = timeit.default_timer() - start_time
        print(elapsed, "seconds taken")

