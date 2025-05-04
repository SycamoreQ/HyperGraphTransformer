import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import ResNet50_Weights , resnet18
import numpy as np 
import seaborn as sns
from torch_geometric.nn import SAGEConv , GATConv , GCNConv
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from encoder import PatchEmbed , LaplacianPositionalEncoding , ConditionalPositionEncoding
from Grapher import Grapher
from torch_geometric.nn import GPSConv

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    

class SelfAttention(nn.Module):
    """
    Multi-head self-attention module
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, embed_dim]
        B, N, C = x.shape
        
        # Calculate query, key, values for all heads
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each has shape [B, num_heads, N, head_dim]
        
        # Calculate attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Apply attention weights to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and feed-forward network
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.attn = SelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.dropout(self.attn(self.norm1(x)))
        
        # MLP with residual connection
        x = x + self.dropout(self.mlp(self.norm2(x)))
        
        return x


class Model_1(nn.Module):
    def __init__(self , num_classes):
        super(Model_1 , self).__init__()
        self.num_classes = num_classes
        self.grouped_conv = nn.Conv2d(in_channels = 16 , out_channels = 32 , kernel_size = 3 , stride = 1 , padding = 1)
        self.transformer_block = nn.TransformerEncoderLayer(d_model = 32 , nhead = 4)
        self.gcn_block = GCNConv(in_channels = 32 , out_channels = 32)
        self.fc = nn.FeedForward(32 , 16)
        
    def forward(self , x , edge_index):
        x = self.grouped_conv(x)
        x = self.transformer_block(x)
        x = self.gcn_block(x , edge_index)
        x = F.relu(x)
        x = F.dropout(x , 0.5)
        x = self.gcn_block(x , edge_index)
        x = F.log_softmax(x , dim = 1)
        return self.fc(x)
    

class Model_2(nn.Module):
    def __init__(self, num_classes, embed_dim=96, depth=12, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super(Model_2, self).__init__()
        self.num_classes = num_classes
        
        self.patch_embed = PatchEmbed()
        self.grapher = Grapher(in_channels=embed_dim)  
        self.laplacian = LaplacianPositionalEncoding(patch_size=4, img_size=224, dim=embed_dim, normalized=True)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.fc = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        B = x.shape[0]
        
        # 1. Patch embedding [B, C, H, W]
        x = self.patch_embed(x)
        
        x = self.grapher(x)

        _, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        
        x = self.laplacian(x)

        for block in self.blocks:
            x = block(x)
        
        x = x.mean(dim=1)  # [B, C]
        
        x = self.fc(x)
        return x

    
class Combined_2(nn.Module):
    def __init__(self , num_classes = 2):
        super(Combined_2 , self).__init__()
        self.cpe = ConditionalPositionEncoding(in_channels = 3 , kernel_size = 3)
        self.gcn = Grapher(in_channels = 3)
        self.fc = nn.Linear(3 , 4)

    def forward(self , x):
        x = self.cpe(x)
        x = self.gcn(x)
        x = x.mean(dim = [2 , 3])
        x = self.fc(x)
        return x

        
    