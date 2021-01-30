sample_rate = 44100     # Target sample rate during feature extraction
window_size = 2048      # Size of FFT window
overlap = 1024          # Amount of overlap between frames
clip_duration = 20     # Duraion of an audio clip (seconds)
seq_len = 860   # Number of frames of an audio clip
#TODO; Need top update seq_len
mel_bins = 128   # Number of Mel bins

# kmax = 3

labels = ['crackles','wheezes']

lb_to_ix = {lb: i for i, lb in enumerate(labels)}
ix_to_lb = {i: lb for i, lb in enumerate(labels)}

from pathlib import Path

class config:
  sr = 44100
  power = 2.0
  mel_bins = 128 
  overlap = 1024
  window_size = 2048
  seq_len = 860


  fmin = 0
  fmax = sr // 2 

  window = 'hann'
  center = True
  pad_mode = 'reflect'
  ref = 1.0
  amin = 1e-10
  top_db = None
  base_dir =  Path('/content/ICBHI_final_database')

  seed=42
  num_folds=4
  workspace = Path('/content/workspace')
  audio_dir =  Path('/content/ICBHI_final_database')
  out_yaml_path = workspace/f'audio.yaml'
  h5_path = workspace/f'development.h5'

  labels = ['crackles','wheezes']

  lb_to_ix = {lb: i for i, lb in enumerate(labels)}
  ix_to_lb = {i: lb for i, lb in enumerate(labels)}

