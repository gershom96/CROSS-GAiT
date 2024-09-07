import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pickle
import numpy as np
from util.datasets import MultimodalDataset
from util.visualize import display_images, plot_curves

import random

def create_weighted_sampler(labels):
    # Count the number of samples in each class
    class_counts = np.bincount(labels)

    print(class_counts)
    
    # Calculate the weights for each sample based on the class frequency
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]
    
    # Create a WeightedRandomSampler to sample based on the calculated weights
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    return sampler

def create_dataloader(h5_file, time_series_mean, time_series_std, image_mean, image_std, batch_size=32):
    # Create the dataset
    print(image_mean)
    dataset = MultimodalDataset(h5_file, time_series_mean, time_series_std, image_mean, image_std)
    
    # Create the weighted sampler
    sampler = create_weighted_sampler(dataset.labels)
    
    # Create the DataLoader with the sampler
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
    
    return dataloader

# Usage
h5_file = '/home/gershom/Documents/GAMMA/ICRA-2025/Outputs/all_data_by_terrain.h5'
batch_size = 32

    
time_series_mean = np.array([-9.7528e-03,  7.1942e-02, -9.8050e+00, -1.3312e-03, -4.1957e-04, -2.5726e-02, 
                    -1.9685e+00,  1.6953e+01, -4.1611e+00,  1.8878e+01, -1.0623e+00,  1.6011e+01, 
                    -3.7153e+00,  1.9238e+01, -1.3624e+01, -1.3926e+01,  1.3605e+01,  1.2424e+01])

time_series_std = np.array([ 1.5537,  0.9568,  3.3148,  0.1399,  0.1181,  0.3017, 
                13.1651, 17.3949, 13.9258, 19.9675, 13.2737, 17.1195, 
                13.4313, 19.9113, 20.9347, 20.9335, 20.7401, 19.3905])
image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]

dataloader = create_dataloader(h5_file, time_series_mean, time_series_std, image_mean, image_std, batch_size)
print_count = 0
# Iterate through the DataLoader
for time_series, images, labels in dataloader:
    print(time_series.shape, images.shape, labels.shape)
    print(labels)

    display_images(images[0],image_mean, image_std)
    plot_curves(time_series[0],time_series_mean, time_series_std)
    print_count +=1

    if(print_count>10):
        break
    
    # You can now use the data for training, testing, etc.