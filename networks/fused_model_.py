import torch.nn as nn

from networks.cross_attention import CrossAttentionBlock
from networks.models_mae import MAEViTEncoder
from networks.causal_cnn import CausalCNNEncoder


import numpy as np
import torch
from functools import partial

class FusionModel(nn.Module):
    def __init__(self, d_k=160, d_v=160, d_embed=768, num_self_attn_layers=7, num_heads=4, projection_dim=64):
        super(FusionModel, self).__init__()
        self.mae_model = MAEViTEncoder(patch_size=16, embed_dim=768, depth=12, num_heads=12,
                                  mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.causal_cnn_encoder = CausalCNNEncoder(in_channels=18, out_channels=160, channels=40, depth=10, reduced_size=320, kernel_size=3)

        for param in self.mae_model.parameters():
            param.requires_grad = False
        for param in self.causal_cnn_encoder.parameters():
            param.requires_grad = False

        self.cross_attention_block = CrossAttentionBlock(d_k=d_k, d_v=d_v, d_embed=d_embed)
        self.self_attention_layers = nn.ModuleList([nn.MultiheadAttention(embed_dim=d_v, num_heads=num_heads, batch_first=True) for _ in range(num_self_attn_layers)])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(d_v) for _ in range(num_self_attn_layers)])
        self.projection_head = nn.Sequential(
            nn.Linear(d_v, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, image, time_series):
        mae_output = self.mae_model(image, mask_ratio=0.75)
        time_series_output = self.causal_cnn_encoder(time_series).unsqueeze(1)
        cross_attention_output = self.cross_attention_block(time_series_output, mae_output, mae_output)
        
        for i, self_attn in enumerate(self.self_attention_layers):
            attn_output, _ = self_attn(cross_attention_output, cross_attention_output, cross_attention_output)
        cross_attention_output = attn_output
        projected_output = self.projection_head(cross_attention_output.squeeze(1))
        return projected_output

class FusionModelWithRegression(nn.Module):
    def __init__(self, d_k=160, d_v=160, d_embed=768, num_self_attn_layers=3, num_heads=4, projection_dim=64, regression_hidden_dim=128):
        super(FusionModelWithRegression, self).__init__()
        self.fusion_model = FusionModel(d_k=d_k, d_v=d_v, d_embed=d_embed, num_self_attn_layers=num_self_attn_layers, num_heads=num_heads, projection_dim=projection_dim)
        

        # checkpoint = torch.load(fusion_model_path)  # Load the trained model checkpoint

        # self.fusion_model.load_state_dict(checkpoint['model_state_dict'])
        # Freeze all FusionModel parameters
        for param in self.fusion_model.parameters():
            param.requires_grad = False
        self.fusion_model.eval()
        
        self.regression_head = RegressionHead(input_dim=projection_dim, hidden_dim=regression_hidden_dim, output_dim=2)  # Output size of 2
        
    def forward(self, image, time_series):
        fused_output = self.fusion_model(image, time_series)
        return self.regression_head(fused_output)
    
class RegressionHead(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=2):
        super(RegressionHead, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim) 
        )
        
    def forward(self, x):
        return self.regressor(x)