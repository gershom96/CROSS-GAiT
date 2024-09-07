import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_k=160, d_v=64, d_embed=768, num_heads=8, dim_feedforward=2048, dropout=0.1, device='cuda:0'):
        super(CrossAttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.device = device  # Store device

        # Linear layers for query, key, and value projections (move to device)
        self.W_q = nn.Linear(d_k, num_heads * d_k, bias=False).to(device)  # For time series (query)
        self.W_k = nn.Linear(d_embed, num_heads * d_k, bias=False).to(device)  # For image (key)
        self.W_v = nn.Linear(d_embed, num_heads * d_v, bias=False).to(device)  # For image (value)
        
        # Output projection after attention
        self.fc_out = nn.Linear(num_heads * d_v, d_k).to(device)

        # Layer Norm and Dropout
        self.norm1 = nn.LayerNorm(d_k).to(device)
        self.dropout1 = nn.Dropout(dropout)

        # Feedforward Network
        self.feedforward = nn.Sequential(
            nn.Linear(d_k, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_k),
        ).to(device)
        self.norm2 = nn.LayerNorm(d_k).to(device)
        self.dropout2 = nn.Dropout(dropout)


    def scaled_dot_product_attention(self, Q, K, V):
        """
        Q: [batch_size, num_heads, query_len, d_k]
        K: [batch_size, num_heads, key_len, d_k]
        V: [batch_size, num_heads, value_len, d_v]
        """
        # Calculate attention scores using scaled dot-product
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32).to(self.device))
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Weighted sum of the values
        output = torch.matmul(attn_weights, V)
        return output

    def forward(self, query, key, value):
        
        batch_size = query.size(0)

        # Ensure that all inputs are on the correct device
        query = query.to(self.device)
        key = key.to(self.device)
        value = value.to(self.device)
        
        # Linear projections for query, key, and value
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, query_len, d_k]
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)    # [batch_size, num_heads, key_len, d_k]
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)  # [batch_size, num_heads, value_len, d_v]
        
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V)  # [batch_size, num_heads, query_len, d_v]
        
        # Concatenate the heads and pass through the final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_v)
        attn_output = self.fc_out(attn_output)  # [batch_size, query_len, d_model]
        
        # Add & Norm
        query = query + self.dropout1(attn_output)  # Residual connection
        query = self.norm1(query)
        
        # Feedforward
        ff_output = self.feedforward(query)
        
        # Add & Norm
        output = query + self.dropout2(ff_output)  # Residual connection
        output = self.norm2(output)
        
        return output
