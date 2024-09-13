import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pickle
import time
import numpy as np
from util.datasets import MultimodalDataset, MultimodalParamDataset
from util.visualize import display_images, plot_curves
from networks.fused_model_ import FusionModelWithRegression
from networks.causal_cnn import CausalCNNEncoder

from networks.models_mae import load_encoder
import random

def create_weighted_sampler_param(dataset):

    class_counts = np.bincount(dataset.sampler_labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in dataset.sampler_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

def create_weighted_sampler(labels):

    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler


def create_dataloader(h5_file, time_series_mean, time_series_std, image_mean, image_std, batch_size=32):
    # Create the dataset
    # print(image_mean)
    # dataset = MultimodalDataset(h5_file, time_series_mean, time_series_std, image_mean, image_std)
    dataset = MultimodalParamDataset(h5_file, time_series_mean, time_series_std, image_mean, image_std)
    # Create the weighted sampler
    sampler = create_weighted_sampler_param(dataset)
    
    # Create the DataLoader with the sampler
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
    
    return dataloader


def load_causal_cnn_encoder(causal_cnn_checkpoint_path, device):
    checkpoint = torch.load(causal_cnn_checkpoint_path, map_location=device)
    causal_cnn_encoder = CausalCNNEncoder(in_channels=18, out_channels=160, channels=40, depth=10, reduced_size=320, kernel_size=3)
    causal_cnn_encoder.load_state_dict(checkpoint)

    for param in causal_cnn_encoder.parameters():
        param.requires_grad = False

    causal_cnn_encoder.to(device)
    causal_cnn_encoder.eval()
    return causal_cnn_encoder

# Usage
h5_file = './Dataset/test_data.h5'
regressor_path = './checkpoints/regressor_checkpoint_epoch_2_step_final.pth'
mae_checkpoint_path = './models/mae_encoder.pth'
causal_cnn_checkpoint_path = './models/encoder_checkpoint_epoch_3_step_3000_CausalCNN_encoder.pth'
fusion_model_path = './checkpoints/model_epoch_100.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# mae_model = load_encoder(mae_checkpoint_path, device)
# causal_cnn_encoder = load_causal_cnn_encoder(causal_cnn_checkpoint_path, device)

batch_size = 1

    
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the model
fusion_model_with_regression = FusionModelWithRegression().to(device)
checkpoint = torch.load(regressor_path)

# Load the state dictionary into the model
fusion_model_with_regression.load_state_dict(checkpoint['model_state_dict'])
fusion_model_with_regression.to(device)
fusion_model_with_regression.eval()

# Iterate through the DataLoader
for time_series, images, labels in dataloader:
    
    print(time_series.shape, images.shape, labels.shape)
    display_images(images[0],image_mean, image_std)
    # display_images(images[1],image_mean, image_std)

    # plot_curves(time_series[0],time_series_mean, time_series_std)

    print_count += 1

    t1 = time.time()
    # Move inputs to the correct device
    time_series = time_series.to(device)
    images = images.to(device)
    # images[0] = images[1]
    # Forward pass
    with torch.no_grad():  # Disable gradient calculation
        output = fusion_model_with_regression(images, time_series)

        t2 = time.time()

        print(t2-t1)
        print(f"Output: {output}")

    if print_count > 10:
        break
