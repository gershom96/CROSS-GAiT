# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import h5py
import numpy as np

import torch
from torch.utils.data import Dataset
import pickle

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


# class MultimodalDataset(Dataset):
#     def __init__(self, pkl_file, time_series_mean, time_series_var, image_mean, image_std):
#         # Load the data from the pickle file
#         with open(pkl_file, 'rb') as f:
#             self.data = pickle.load(f)
        
#         self.samples = []
#         self.labels = []
        
#         # Store the provided means and variances for normalization
#         self.time_series_mean = torch.tensor(time_series_mean, dtype=torch.float32).view(-1, 1)  # Reshape to [channels, 1]
#         self.time_series_std = torch.sqrt(torch.tensor(time_series_var, dtype=torch.float32)).view(-1, 1)
        
#         self.image_mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)  # For RGB channels
#         self.image_std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)
        
#         # Flatten the data and create a list of samples and their corresponding labels
#         for label, class_data in self.data.items():
#             for sample in class_data:
#                 self.samples.append(sample)  # Each sample is a tuple (time_series_data, image_data)
#                 self.labels.append(int(label))  # Convert the label to an integer
        
#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, idx):
#         time_series_data, image_data = self.samples[idx]
#         label = self.labels[idx]
        
#         # Convert the data to tensors
#         time_series_tensor = torch.tensor(time_series_data, dtype=torch.float32)
#         image_tensor = torch.tensor(image_data, dtype=torch.float32).permute(2, 0, 1)  # Convert to CxHxW
        
#         # Normalize the time series data
#         time_series_tensor = (time_series_tensor - self.time_series_mean) / self.time_series_std
        
#         # Normalize the image data
#         image_tensor = (image_tensor - self.image_mean) / self.image_std
        
#         return time_series_tensor, image_tensor, label

class MultimodalDataset(Dataset):
    def __init__(self, h5_file, time_series_mean, time_series_std, image_mean, image_std):
        # Open the HDF5 file for reading
        self.h5_file = h5_file

        # Store the provided means and standard deviations for normalization
        self.time_series_mean = torch.tensor(time_series_mean, dtype=torch.float32).view(-1, 1)  # Reshape to [channels, 1]
        self.time_series_std = torch.tensor(time_series_std, dtype=torch.float32).view(-1, 1)
        
        self.image_mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)  # For RGB channels
        self.image_std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)

        self.samples = []
        self.labels = []

        # Load the data from the HDF5 file
        with h5py.File(self.h5_file, 'r') as hf:
            # Loop through each terrain (0, 1, 2, 3, 4)
            for terrain_label in hf.keys():
                group = hf[terrain_label]
                time_series_data = group['time_series'][:]
                image_data = group['image'][:]

                # Store the data with corresponding labels
                for i in range(time_series_data.shape[0]):
                    self.samples.append([time_series_data[i], image_data[i]])
                    self.labels.append(int(terrain_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        time_series_data, image_data = self.samples[idx]
        label = self.labels[idx]
        
        # Convert the data to tensors
        time_series_tensor = torch.tensor(time_series_data, dtype=torch.float32)
        image_tensor = torch.tensor(image_data, dtype=torch.float32).permute(2, 0, 1)/ 255.0  # Convert to CxHxW
        
        # Normalize the time series data
        time_series_tensor = (time_series_tensor - self.time_series_mean) / self.time_series_std
        
        # Normalize the image data
        image_tensor = (image_tensor - self.image_mean) / self.image_std
        
        return time_series_tensor, image_tensor, label