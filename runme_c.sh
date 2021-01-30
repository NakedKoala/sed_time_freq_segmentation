#!/bin/bash


# DCASE2018_TASK2_DATASET_DIR="/content/FSDKaggle2018"
# DCASE2018_TASK1_DATASET_DIR="/content/TUT-urban-acoustic-scenes-2018-development"

WORKSPACE="/content/workspace/"


# Create YAML
python3 create_yaml_c.py


# Calculate features
python3 features_c.py

# Train
MODEL_TYPE="gwrp"    # 'gmp' | 'gap' | 'gwrp'
HOLDOUT_FOLD=1
SNR=0
CUDA_VISIBLE_DEVICES=0 python3 pytorch/main_pytorch.py train --workspace=$WORKSPACE --model_type=$MODEL_TYPE --scene_type=dcase2018_task1 --snr=$SNR --holdout_fold=$HOLDOUT_FOLD --cuda

# # Inference
# CUDA_VISIBLE_DEVICES=0 python3 pytorch/main_pytorch.py inference --workspace=$WORKSPACE --model_type=$MODEL_TYPE --scene_type=dcase2018_task1 --snr=$SNR --holdout_fold=$HOLDOUT_FOLD --iteration=10000 --cuda

# # Get average statistics
# python3 utils/get_avg_stats.py single_fold --workspace=$WORKSPACE --filename=main_pytorch --model_type=$MODEL_TYPE --scene_type=dcase2018_task1 --snr=$SNR --holdout_fold=$HOLDOUT_FOLD 

# # After train & inference and calculate the statistics of folds 1, 2, 3 and 4, you may run the following command to get averaged statistics of all folds. 
# python3 utils/get_avg_stats.py all_fold --workspace=$WORKSPACE --filename=main_pytorch --model_type=$MODEL_TYPE --scene_type=dcase2018_task1 --snr=$SNR


# ############# Plot figures for paper #############
# # Visualize waveform & spectrogram
# python3 utils/visualize.py waveform --workspace=$WORKSPACE --scene_type=dcase2018_task1 --snr=$SNR --holdout_fold=$HOLDOUT_FOLD --cuda

# # Visualze learned segmentation masks & SED results
# CUDA_VISIBLE_DEVICES=0 python3 utils/visualize.py mel_masks --workspace=$WORKSPACE --model_type=$MODEL_TYPE --scene_type=dcase2018_task1 --snr=$SNR --holdout_fold=$HOLDOUT_FOLD --iteration=10000 --cuda
