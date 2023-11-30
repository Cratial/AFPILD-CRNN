"""
    Dataloader for AFPILD dataset.
"""

from torchvision import datasets
import torchvision.transforms as tv_transforms
from base import BaseDataLoader
import torch.utils.data as data
import librosa
import os
import pandas as pd
import torch
import numpy as np
import pathlib
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, TimeMask
from utils.spec_timeshift_transform import TimeShift
from sklearn.model_selection import train_test_split


def load_dataframe(path):
    df = pd.read_csv(path)
    return df.to_dict('records')


def load_numpy(path):
    try:
        data = np.load(path)
    except Exception as e:
        print(e)
        return None
    return data


class AFPILD_spec_gcc_Dataset(data.Dataset):
    def __init__(self, dataset_dir, task, data_type='train', spec_shift=False, gcc_shift=False):
        task_type, train_type, covariant_type = task.split('_')

        # anastrophic setting
        if train_type == 'ana':
            if data_type == 'train':
                data_type = 'test'
            elif data_type == 'test':
                data_type = 'train'

        self.task = task_type
        meta_file = os.path.join(dataset_dir, f"AFPILD_FE1_{covariant_type+'_'+data_type}.csv")
        self.data_arr = load_dataframe(meta_file)

        self.data_dir = dataset_dir
        self.spec_shift = spec_shift
        self.gcc_shift = gcc_shift

        # spectrogram augmentation
        spec_transforms = []
        if spec_shift:
            spec_transforms.append(TimeShift())
        if len(spec_transforms) == 0:
            self.spec_transform = None
        else:
            self.spec_transform = tv_transforms.Compose(spec_transforms)

        # gcc-phat augmentation
        gcc_transforms = []
        if gcc_shift:
            gcc_transforms.append(TimeShift())
        if len(gcc_transforms) == 0:
            self.gcc_transform = None
        else:
            self.gcc_transform = tv_transforms.Compose(gcc_transforms)

    def __len__(self):
        return len(self.data_arr)

    def __getitem__(self, idx):
        item = self.data_arr[idx]
        sub_label = item['subject_label']
        sub_label_encoded = int(sub_label[1:]) - 1  # subject label for person classification
        loc_theta = item['loc_azimuth']    # [-180, 180]

        # load audio feature
        spec_fea = load_numpy(os.path.join(self.data_dir, item['fea_spec']))  # (freq_dim x time_dim x channel_num)
        gcc_fea = load_numpy(os.path.join(self.data_dir, item['fea_gcc']))    # (freq_dim x time_dim x channel_num)

        # Add transforms
        if self.spec_transform is not None:
            spec_fea = self.spec_transform(spec_fea)

        if self.gcc_transform is not None:
            gcc_fea = self.gcc_transform(gcc_fea)

        # return spec, scene_encoded, theta belong to [-180, 180]
        # return {"spec": spec_fea, "gcc": gcc_fea}, loc_theta, sub_label_encoded

        # normalized azimuth
        # return {"spec": spec_fea, "gcc": gcc_fea}, loc_theta/180.0, sub_label_encoded
        if self.task == 'ide':
            return {"spec": spec_fea, "gcc": gcc_fea}, sub_label_encoded

        elif self.task == 'loc':
            return {"spec": spec_fea, "gcc": gcc_fea}, np.array(loc_theta/180.0)

        elif self.task == 'ideloc':
            # concatenate loc_theta & sub_label
            return {"spec": spec_fea, "gcc": gcc_fea}, np.array([sub_label_encoded, loc_theta / 180.0])

        elif self.task == 'accil':
            # for ACCDOA representation
            se_label = np.zeros(40)
            x_label = np.zeros(40)
            y_label = np.zeros(40)

            se_label[sub_label_encoded] = 1
            x_label[sub_label_encoded] = np.cos(loc_theta*np.pi/180.0)
            y_label[sub_label_encoded] = np.sin(loc_theta*np.pi/180.0)

            x_label = se_label * x_label
            y_label = se_label * y_label

            accdoa_xy = np.concatenate((x_label, y_label), axis=0)

            return {"spec": spec_fea, "gcc": gcc_fea}, accdoa_xy

        else:
            print('!!! Task setting error !!!!')

