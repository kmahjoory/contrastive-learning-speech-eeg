


import os, sys, glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np

import mne

import matplotlib
import matplotlib.pyplot as plt
import time

from src.train import eval_model_cl
from src.data.utils import unfold_raw, rm_repeated_annotations



def load_data(subject_ids, subjects_dir, window_size, stride_size_train, stride_size_val, stride_size_test, n_channs):

    raws_train_windowed, raws_val_windowed, raws_test_windowed = [], [], []

    for subj_id in subject_ids:
        

        # load subject raw MNE object
        raw = mne.io.read_raw(os.path.join(subjects_dir, f'subj_{subj_id}_after_ica_raw.fif'), preload=True)
        # drop M1 and M2 channels
        raw.drop_channels(['M1', 'M2'])
        assert raw.info['nchan'] == n_channs

        raw = rm_repeated_annotations(raw)
        annots = raw.annotations.copy()
        raw_split = [raw.copy().crop(t1, t2) for t1, t2 in zip(annots.onset[:-1]+annots.duration[:-1], annots.onset[1:])]

        # Pick the split with the longest duration for validation, supposedly less noisy
        ix_val = np.argmax([i.get_data().shape[1] for i in raw_split])
        raw_val = [raw_split.pop(ix_val)] # create a list to make it iterable. later may be used for multiple splits

        # Pick the next split with the longest duration for testing, supposedly less noisy
        ix_test = np.argmax([i.get_data().shape[1] for i in raw_split])
        raw_test = [raw_split.pop(ix_test)]
        
        # creat list of unfolded tensor raw objects
        fs = raw.info['sfreq']
        raws_train_windowed.extend([unfold_raw(i, window_size=window_size, stride=stride_size_train) for i in raw_split if i.get_data().shape[1] > window_size])
        raws_val_windowed.extend([unfold_raw(i, window_size=window_size, stride=stride_size_val) for i in raw_val if i.get_data().shape[1] > window_size])
        raws_test_windowed.extend([unfold_raw(i, window_size=window_size, stride=stride_size_test) for i in raw_test if i.get_data().shape[1] > window_size])
        print("-------------------------------------")
        print('N train: %d  N val: %d  N test: %d' % (len(raws_train_windowed), len(raws_val_windowed), len(raws_test_windowed)))

    # concatenate all in second dimension
    sigs_train = torch.cat(raws_train_windowed, dim=1).permute(1, 0, 2, 3)
    sigs_val = torch.cat(raws_val_windowed, dim=1).permute(1, 0, 2, 3)
    sigs_test = torch.cat(raws_test_windowed, dim=1).permute(1, 0, 2, 3)
    print(f"Shape Trian: {sigs_train.shape}  Shape Val: {sigs_val.shape}  Shape Test: {sigs_test.shape}")

    eegs_train = sigs_train[:, :, :-1, :]
    eegs_val = sigs_val[:, :, :-1, :]
    eegs_test = sigs_test[:, :, :-1, :]
    print("-------------------------------------")
    print(f"Shape EEG Train: {eegs_train.shape}  Val: {eegs_val.shape}  Test: {eegs_test.shape}")

    # To avoid information leakage, we estimate the mean and std from the training set only.
    mean_eeg_train =  eegs_train.mean()
    std_eeg_train = eegs_train.std()
    print(f"Mean: {mean_eeg_train}  Std: {std_eeg_train}")

    envs_train = sigs_train[:, :, [-1], :]
    envs_val = sigs_val[:, :, [-1], :]
    envs_test = sigs_test[:, :, [-1], :]
    print(f"Shape Env Train: {envs_train.shape}  Val: {envs_val.shape}  Test: {envs_test.shape}")

    # Estimate mean and std of the Envelope data set
    mean_env_train =  envs_train.mean()
    std_env_train = envs_train.std()
    print(f"Mean Env: {mean_env_train}  Std Env: {std_env_train}")

    # Normalize the data
    eegs_train = (eegs_train - mean_eeg_train) / std_eeg_train
    eegs_val = (eegs_val - mean_eeg_train) / std_eeg_train
    eegs_test = (eegs_test - mean_eeg_train) / std_eeg_train

    envs_train = (envs_train - mean_env_train) / std_env_train
    envs_val = (envs_val - mean_env_train) / std_env_train
    envs_test = (envs_test - mean_env_train) / std_env_train

    return {'eegs_train': eegs_train, 'eegs_val': eegs_val, 'eegs_test': eegs_test, 
            'envs_train': envs_train, 'envs_val': envs_val, 'envs_test': envs_test}



def load_data_sa(subject_ids, subjects_dir, window_size, stride_size_train, 
                 stride_size_val=None, stride_size_test=None, n_channs=130, 
                 val_split=False, test_split=False, normalize_data=True):

    raws_train_windowed, raws_val_windowed, raws_test_windowed = [], [], []

    for subj_id in subject_ids:
        
        # load subject raw MNE object
        raw = mne.io.read_raw(os.path.join(subjects_dir, f'subj_{subj_id}_after_ica_raw.fif'), preload=True)
        # drop M1 and M2 channels
        print(raw.info['nchan'])
        raw.drop_channels(['M1', 'M2'])
        assert raw.info['nchan'] == n_channs

        raw = rm_repeated_annotations(raw)
        annots = raw.annotations.copy()
        raw_split = [raw.copy().crop(t1, t2) for t1, t2 in zip(annots.onset[:-1]+annots.duration[:-1], annots.onset[1:])]

        # Pick the split with the longest duration for validation, supposedly less noisy
        if val_split:
            ix_val = np.argmax([i.get_data().shape[1] for i in raw_split])
            raw_val = [raw_split.pop(ix_val)] # create a list to make it iterable. later may be used for multiple splits
            raws_val_windowed.extend([unfold_raw(i, window_size=window_size, stride=stride_size_val) for i in raw_val if i.get_data().shape[1] > window_size])
        
        # Pick the next split with the longest duration for testing, supposedly less noisy
        if test_split:
            ix_test = np.argmax([i.get_data().shape[1] for i in raw_split])
            raw_test = [raw_split.pop(ix_test)]        
            raws_test_windowed.extend([unfold_raw(i, window_size=window_size, stride=stride_size_test) for i in raw_test if i.get_data().shape[1] > window_size])
        
        # creat list of unfolded tensor raw objects
        fs = raw.info['sfreq']
        raws_train_windowed.extend([unfold_raw(i, window_size=window_size, stride=stride_size_train) for i in raw_split if i.get_data().shape[1] > window_size])

        print(f"----------------  Subject: {subj_id} loaded. --------------")
        print('N train: %d  N val: %d  N test: %d' % (len(raws_train_windowed), len(raws_val_windowed), len(raws_test_windowed)))

    # concatenate all in second dimension
    sigs_train = torch.cat(raws_train_windowed, dim=1).permute(1, 0, 2, 3)
    eegs_train = sigs_train[:, :, :-2, :]
    envs_train = sigs_train[:, :, -2:, :]
    out = {'eegs_train': eegs_train, 'envs_train': envs_train}

    if normalize_data==True:
        # To avoid information leakage, we estimate the mean and std from the training set only.
        mean_eeg_train =  eegs_train.mean()
        std_eeg_train = eegs_train.std()
        mean_env_train =  envs_train.mean()
        std_env_train = envs_train.std()
        # Normalize the data
        eegs_train = (eegs_train - mean_eeg_train) / std_eeg_train
        envs_train = (envs_train - mean_env_train) / std_env_train
        out.update({'mean_eegs_train': mean_eeg_train, 'std_eegs_train': std_eeg_train,
                    'mean_envs_train': mean_env_train, 'std_envs_train': std_env_train})
    print('-------------------------------------------------')
    print(f"Shape EEG Train: {eegs_train.shape}    Shape Env Train: {envs_train.shape}")

    if val_split:
        sigs_val = torch.cat(raws_val_windowed, dim=1).permute(1, 0, 2, 3)
        eegs_val = sigs_val[:, :, :-2, :]
        eegs_val = (eegs_val - mean_eeg_train) / std_eeg_train
        envs_val = sigs_val[:, :, -2:, :]
        envs_val = (envs_val - mean_env_train) / std_env_train
        out.update({'eegs_val': eegs_val, 'envs_val': envs_val})
        print(f"Shape EEG Train: {eegs_val.shape}    Shape Env Train: {envs_val.shape}")

    if test_split:
        sigs_test = torch.cat(raws_test_windowed, dim=1).permute(1, 0, 2, 3)
        eegs_test = sigs_test[:, :, :-2, :]    
        eegs_test = (eegs_test - mean_eeg_train) / std_eeg_train
        envs_test = sigs_test[:, :, -2:, :]
        envs_test = (envs_test - mean_env_train) / std_env_train
        out.update({'eegs_test': eegs_test, 'envs_test': envs_test})
        print(f"Shape EEG Train: {eegs_test.shape}    Shape Env Train: {envs_test.shape}")
    
    return out
