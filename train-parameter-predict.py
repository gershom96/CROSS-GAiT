import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from networks.fused_model_ import FusionModelWithRegression
from networks.causal_cnn import CausalCNNEncoder
from networks.models_mae import load_encoder

from util.datasets import MultimodalParamDataset

import numpy as np

def load_causal_cnn_encoder(causal_cnn_checkpoint_path, device):
    checkpoint = torch.load(causal_cnn_checkpoint_path, map_location=device)
    causal_cnn_encoder = CausalCNNEncoder(in_channels=18, out_channels=160, channels=40, depth=10, reduced_size=320, kernel_size=3)
    causal_cnn_encoder.load_state_dict(checkpoint)

    for param in causal_cnn_encoder.parameters():
        param.requires_grad = False

    causal_cnn_encoder.to(device)
    causal_cnn_encoder.eval()
    return causal_cnn_encoder

def create_weighted_sampler_param(dataset):

    class_counts = np.bincount(dataset.sampler_labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in dataset.sampler_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler
# Define your loss function and optimizer for regression
def train_model(fusion_model_with_regression, dataloader, device, epochs=20, lr=1e-4):

   
    # Define the Mean Squared Error (MSE) Loss for regression
    criterion = nn.MSELoss()
    
    # Adam optimizer for training
    optimizer = optim.AdamW(fusion_model_with_regression.parameters(), lr=1e-4, weight_decay=1e-2)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        fusion_model_with_regression.train()  # Set the model to training mode
        
        for batch_idx, (time_series_data, image_data, target_params) in enumerate(dataloader):
            time_series_data, image_data, target_params = time_series_data.to(device), image_data.to(device), target_params.to(device)
            
            optimizer.zero_grad()
            predictions = fusion_model_with_regression(image_data, time_series_data)
            
            # Compute the regression loss
            loss = criterion(predictions, target_params)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            print(loss.item())
            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

# mae_checkpoint_path = './models/mae_encoder.pth'
# causal_cnn_checkpoint_path = './models/encoder_checkpoint_epoch_3_step_3000_CausalCNN_encoder.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_path = "./Dataset/test_data.h5"


time_series_mean = np.array([-9.7528e-03,  7.1942e-02, -9.8050e+00, -1.3312e-03, -4.1957e-04, -2.5726e-02, 
                    -1.9685e+00,  1.6953e+01, -4.1611e+00,  1.8878e+01, -1.0623e+00,  1.6011e+01, 
                    -3.7153e+00,  1.9238e+01, -1.3624e+01, -1.3926e+01,  1.3605e+01,  1.2424e+01])

time_series_std = np.array([ 1.5537,  0.9568,  3.3148,  0.1399,  0.1181,  0.3017, 
                13.1651, 17.3949, 13.9258, 19.9675, 13.2737, 17.1195, 
                13.4313, 19.9113, 20.9347, 20.9335, 20.7401, 19.3905])
image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]

# Example initialization
# mae_model = load_encoder(mae_checkpoint_path, device)
# causal_cnn_encoder = load_causal_cnn_encoder(causal_cnn_checkpoint_path, device)

batch_size = 32
train_dataset = MultimodalParamDataset(dataset_path, time_series_mean, time_series_std, image_mean, image_std)
train_sampler = create_weighted_sampler_param(train_dataset)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
fusion_model_with_regression = FusionModelWithRegression().to(device)

# Assuming `dataloader` is already created and provides time_series_data, image_data, and target_params (hip splay, leg height).
train_model(fusion_model_with_regression, train_dataloader, device, epochs=20)
