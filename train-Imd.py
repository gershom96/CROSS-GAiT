import torch
import torch.nn as nn
import time
import timm
from networks.causal_cnn import CausalCNNEncoder
from networks.models_mae import load_encoder
from networks.cross_attention import CrossAttentionBlock

# Load the CausalCNN Time Series Encoder
# def load_causal_cnn_encoder(causal_cnn_checkpoint_path, device):
#     causal_cnn_encoder = CausalCNNEncoder(in_channels=18, out_channels=320, channels=40, depth=10, reduced_size=160, kernel_size=3)
#     causal_cnn_encoder.load_state_dict(torch.load(causal_cnn_checkpoint_path))
    
#     # Freeze Causal CNN Encoder
#     for param in causal_cnn_encoder.parameters():
#         param.requires_grad = False

#     causal_cnn_encoder.to(device)
#     causal_cnn_encoder.eval()
    
#     return causal_cnn_encoder
def load_causal_cnn_encoder(causal_cnn_checkpoint_path, device):
    # Load the causal CNN model and map to the current available device
    checkpoint = torch.load(causal_cnn_checkpoint_path, map_location=device)
    causal_cnn_encoder = CausalCNNEncoder(in_channels=18, out_channels=160, channels=40, depth=10, reduced_size=320, kernel_size=3)  # Replace with your model initialization
    causal_cnn_encoder.load_state_dict(checkpoint)

    # Freeze Causal CNN Encoder
    for param in causal_cnn_encoder.parameters():
        param.requires_grad = False

    causal_cnn_encoder.to(device)
    causal_cnn_encoder.eval()
    return causal_cnn_encoder

# Forward pass through both encoders
def forward_encoders(mae_model, causal_cnn_encoder, image, time_series_data, device):
    # Pass image through MAE encoder (ViT-based model)
    with torch.no_grad():
        # Assuming `image` is a batch of images with shape [batch_size, 3, 224, 224]
        mae_output = mae_model(image.to(device), mask_ratio=0.75)  # Call the `forward()` method instead of `forward_features`

    # Pass time series data through CausalCNN encoder
    with torch.no_grad():
        # Assuming `time_series_data` is [batch_size, 18, 100] or similar
        time_series_output = causal_cnn_encoder(time_series_data.to(device))  # [batch_size, 160]

    return mae_output, time_series_output

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Define the paths to your model checkpoints
    mae_checkpoint_path = './models/mae_encoder.pth'
    causal_cnn_checkpoint_path = './models/encoder_checkpoint_epoch_1_step_600_CausalCNN_encoder.pth'

    # Load encoders
    mae_model = load_encoder(mae_checkpoint_path, device)
    causal_cnn_encoder = load_causal_cnn_encoder(causal_cnn_checkpoint_path, device)
    
    cross_attention_block = CrossAttentionBlock(d_k = 160, d_v = 64, d_embed = 768)
    # Example data (replace with your actual data loaders)
    example_image = torch.randn(2, 3, 224, 224)  # Batch of 8 images
    example_time_series = torch.randn(2, 18, 100)  # Batch of 8 time-series data

    mae_output, causal_cnn_output = forward_encoders(mae_model, causal_cnn_encoder, example_image, example_time_series, device)
    causal_cnn_output = causal_cnn_output.unsqueeze(1)

    cross_attention_output = cross_attention_block(causal_cnn_output, mae_output, mae_output)
    
    # for i in range(100):
        # t1 = time.time()
        # Forward pass to get outputs from both models
        # mae_output, causal_cnn_output = forward_encoders(mae_model, causal_cnn_encoder, example_image, example_time_series, device)
        
        # t2 = time.time()

        # print(t2-t1)
    # Output dimensions
    # print(f'MAE Output: {mae_output.shape}, Causal CNN Output: {causal_cnn_output.shape}')

    print(f'MAE Output: {mae_output.shape}, Causal CNN Output: {causal_cnn_output.shape}, Cross Attention Output: {cross_attention_output.shape}')

if __name__ == "__main__":
    main()
