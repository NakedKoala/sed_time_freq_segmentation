import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))
import numpy as np
import pandas as pd
import argparse
import h5py
import librosa
from scipy import signal
import matplotlib.pyplot as plt
import time
import csv
import random
import yaml
import logging

from utilities import read_audio, create_folder, get_filename, create_logging
import config
from pathlib import Path
from config_c import config


class LogMelExtractor():
    def __init__(self, sample_rate, window_size, overlap, mel_bins):
        
        self.window_size = window_size
        self.overlap = overlap
        self.ham_win = np.hamming(window_size)
        
        self.melW = librosa.filters.mel(sr=sample_rate, 
                                        n_fft=window_size, 
                                        n_mels=mel_bins, 
                                        fmin=50., 
                                        fmax=sample_rate // 2).T
        '''(fft_size, mel_bins)'''
    
    def transform(self, audio):
    
        x = self.transform_stft(audio)
    
        x = np.dot(x, self.melW)
        x = np.log(x + 1e-8)
        x = x.astype(np.float32)
        
        return x
        
    def transform_stft(self, audio):
        
        ham_win = self.ham_win
        window_size = self.window_size
        overlap = self.overlap
    
        [f, t, x] = signal.spectral.spectrogram(
                        audio, 
                        window=ham_win, 
                        nperseg=window_size, 
                        noverlap=overlap, 
                        detrend=False, 
                        return_onesided=True, 
                        mode='magnitude')
        
        x = x.T
        x = x.astype(np.float32)
        
        return x
        
    def get_inverse_melW(self):
        """Transformation matrix for convert back from mel bins to stft bins. 
        """
        
        W = self.melW.T     # (mel_bins, fft_size)
        invW = W / (np.sum(W, axis=0) + 1e-8)
        return invW


def calculate_logmel(audio_path, sample_rate, feature_extractor):
    
    # Read audio
    (audio, fs) = read_audio(audio_path, target_fs=sample_rate, mono=True)
    
    # Extract feature
    logmel = feature_extractor.transform(audio)

 
    dict = {'logmel': logmel}
    
    return dict

def get_target_from_events(events, lb_to_ix):
    
    classes_num = len(lb_to_ix)
    target = np.zeros(classes_num, dtype=np.int32)
    
    for event in events:
        ix = lb_to_ix[event['event_label']]
        target[ix] = 1
        
    return target


def calculate_logmel_features(config):
    
    # Arguments & parameters
    workspace = config.workspace

    sample_rate = config.sr
    window_size = config.window_size
    overlap = config.overlap
    seq_len = config.seq_len
    mel_bins = config.mel_bins
    stft_bins = window_size // 2 + 1
    classes_num = len(config.labels)
    lb_to_ix = config.lb_to_ix
    
    # Paths
    audio_dir = config.audio_dir

    yaml_path = config.out_yaml_path

    hdf5_path = config.h5_path
        
    create_folder(hdf5_path.parents[0])

    # # Load  yaml
    load_time = time.time()
    
    with open(yaml_path, 'r') as f:
        data_list = yaml.load(f)
        
    logging.info('Loading yaml time: {} s'
        ''.format(time.time() - load_time))
    
    # Feature extractor
    feature_extractor = LogMelExtractor(sample_rate=sample_rate, 
                                        window_size=window_size, 
                                        overlap=overlap, 
                                        mel_bins=mel_bins)

    # Create hdf5 file
    write_hdf5_time = time.time()
    
    hf = h5py.File(hdf5_path, 'w')
    
    hf.create_dataset(
        name='logmel', 
        shape=(0, seq_len, mel_bins), 
        maxshape=(None, seq_len, mel_bins), 
        dtype=np.float32)
                
    hf.create_dataset(
        name='target', 
        shape=(0, classes_num), 
        maxshape=(None, classes_num), 
        dtype=np.int32)
        
    audio_names = []
   
    folds = []

    for n, data in enumerate(data_list):
        
        if n % 10 == 0:
            logging.info('{} / {} audio features calculated'
                ''.format(n, len(data_list)))
            
        audio_path = audio_dir/f'{data["fname"]}'
    
        audio_names.append(data['fname'])
        folds.append(data['fold'])
    
        # Extract feature
        features_dict = calculate_logmel(audio_path=audio_path, 
                                         sample_rate=config.sr, 
                                         feature_extractor=feature_extractor)
    
        # Write out features
        hf['logmel'].resize((n + 1, seq_len, mel_bins))
        hf['logmel'][n] = features_dict['logmel']

      
        # Write out target
        target = get_target_from_events(data['events'], lb_to_ix)
        hf['target'].resize((n + 1, classes_num))
        hf['target'][n] = target
        
    hf.create_dataset(name='audio_name', 
                      data=[s.encode() for s in audio_names], 
                      dtype='S40')
                        
    hf.create_dataset(name='fold', 
                      data=folds, 
                      dtype=np.int32)

    hf.close()
    
    logging.info('Write out hdf5 file to {}'.format(hdf5_path))
    logging.info('Time spent: {} s'.format(time.time() - write_hdf5_time))


if __name__ == '__main__':
  
        
    logs_dir = os.path.join(config.workspace, 'logs', get_filename(__file__))
    create_folder(logs_dir)
    logging = create_logging(logs_dir, filemode='w')
    
    logging.info(config)

    calculate_logmel_features(config)
        
