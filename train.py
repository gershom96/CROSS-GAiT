import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.optim as optim
import wandb

from util.datasets import MultimodalDataset
from networks.causal_cnn import CausalCNNEncoder
from networks.models_mae import load_encoder
from networks.fused_model import FusionModel

from losses.sup_con_loss import SupConLoss
import numpy as np
import os


def create_weighted_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler


def load_causal_cnn_encoder(causal_cnn_checkpoint_path, device):
    checkpoint = torch.load(causal_cnn_checkpoint_path, map_location=device)
    causal_cnn_encoder = CausalCNNEncoder(in_channels=18, out_channels=160, channels=40, depth=10, reduced_size=320, kernel_size=3)
    causal_cnn_encoder.load_state_dict(checkpoint)

    for param in causal_cnn_encoder.parameters():
        param.requires_grad = False

    causal_cnn_encoder.to(device)
    causal_cnn_encoder.eval()
    return causal_cnn_encoder

def save_checkpoint(epoch, model, optimizer, loss, checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pth')
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, checkpoint_path)
    
    print(f"Checkpoint saved at {checkpoint_path}")


def main():
    # Initialize wandb
    wandb.init(project="fusion_model_training")

    batch_size = 256
    num_epochs = 100

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mae_checkpoint_path = './models/mae_encoder.pth'
    causal_cnn_checkpoint_path = './models/encoder_checkpoint_epoch_1_step_600_CausalCNN_encoder.pth'

    mae_model = load_encoder(mae_checkpoint_path, device)
    causal_cnn_encoder = load_causal_cnn_encoder(causal_cnn_checkpoint_path, device)
    
    fusion_model = FusionModel(mae_model, causal_cnn_encoder, d_k=160, d_v=160, d_embed=768, num_self_attn_layers=3, num_heads=4).to(device)

    criterion = SupConLoss(temperature=0.07).to(device)
    optimizer = optim.AdamW(fusion_model.parameters(), lr=1e-4, weight_decay=1e-2)

    h5_file = '/home/gershom/Documents/GAMMA/ICRA-2025/Outputs/all_data_by_terrain.h5'
    
    time_series_mean = np.array([-9.7528e-03,  7.1942e-02, -9.8050e+00, -1.3312e-03, -4.1957e-04, -2.5726e-02, 
                     -1.9685e+00,  1.6953e+01, -4.1611e+00,  1.8878e+01, -1.0623e+00,  1.6011e+01, 
                     -3.7153e+00,  1.9238e+01, -1.3624e+01, -1.3926e+01,  1.3605e+01,  1.2424e+01])
    
    time_series_std = np.array([ 1.5537,  0.9568,  3.3148,  0.1399,  0.1181,  0.3017, 
                    13.1651, 17.3949, 13.9258, 19.9675, 13.2737, 17.1195, 
                    13.4313, 19.9113, 20.9347, 20.9335, 20.7401, 19.3905])
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]

    dataset = MultimodalDataset(h5_file, time_series_mean, time_series_std, image_mean, image_std)
    sampler = create_weighted_sampler(dataset.labels)
    dataloader = DataLoader(dataset, batch_size, sampler=sampler, num_workers=4)

    fusion_model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_idx, (time_series_data, image_data, labels) in enumerate(dataloader):
            time_series_data, image_data, labels = time_series_data.to(device), image_data.to(device), labels.to(device)
            optimizer.zero_grad()
            projections = fusion_model(image_data, time_series_data)
            projections = projections.unsqueeze(1)
            loss = criterion(projections, labels)
            loss.backward()
            optimizer.step()

            # Print step loss
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.4f}")
        
        # Log epoch loss to wandb
        wandb.log({"Epoch": epoch + 1, "Loss": avg_epoch_loss})
        
        # Save checkpoint every epoch

        if ( epoch+1 % 10 == 0 or epoch == num_epochs-1):
            save_checkpoint(epoch + 1, fusion_model, optimizer, avg_epoch_loss)

    wandb.finish()


if __name__ == "__main__":
    main()