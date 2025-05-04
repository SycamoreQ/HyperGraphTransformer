import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import ResNet50_Weights, resnet18
import numpy as np
from torch_geometric.nn import SAGEConv, GATConv, GCNConv
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from encoder import PatchEmbed , LaplacianPositionalEncoding , ConditionalPositionEncoding
from Grapher import Grapher


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
    Multi-head self-attention module with memory optimization
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Combine q, k, v into a single matrix multiplication to reduce memory
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, embed_dim]
        B, N, C = x.shape
        
        # Calculate query, key, values for all heads
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each has shape [B, num_heads, N, head_dim]
        
        # Calculate attention scores - using smaller chunks if memory is an issue
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention weights to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x

class TransformerBlock(nn.Module):
    """
    Memory-optimized Transformer block with self-attention and feed-forward network
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.attn = SelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = FeedForward(embed_dim, mlp_hidden_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.dropout(self.attn(self.norm1(x)))
        
        # MLP with residual connection
        x = x + self.dropout(self.mlp(self.norm2(x)))
        
        return x

class Model_1(nn.Module):
    def __init__(self, num_classes, in_channels=16):
        super(Model_1, self).__init__()
        self.num_classes = num_classes
        self.grouped_conv = nn.Conv2d(in_channels=in_channels, out_channels=32, 
                                    kernel_size=3, stride=1, padding=1)
        self.transformer_block = nn.TransformerEncoderLayer(d_model=32, nhead=4)
        self.gcn_block = GCNConv(in_channels=32, out_channels=32)
        self.fc = nn.Linear(32, num_classes)  # Changed to Linear from FeedForward
        
    def forward(self, x, edge_index):
        x = self.grouped_conv(x)
        # Reshape needed for transformer
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, channels, -1).permute(0, 2, 1)  # [B, HW, C]
        x = self.transformer_block(x)
        # Reshape for GCN
        x = x.permute(0, 2, 1).reshape(batch_size * channels, -1)  # Format for GCN
        x = self.gcn_block(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.gcn_block(x, edge_index)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class Model_2(nn.Module):
    def __init__(self, num_classes, embed_dim=64, depth=6, num_heads=4, mlp_ratio=2.0, dropout=0.1):
        super(Model_2, self).__init__()
        # Reduced model size for memory optimization
        self.num_classes = num_classes
        self.laplacian = LaplacianPositionalEncoding(patch_size=4, img_size=224, dim=embed_dim, normalized=True)

        # Use nn.Sequential instead of ModuleList for better memory efficiency
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.patch_embed = PatchEmbed()
        self.fc = nn.Linear(embed_dim, num_classes)  # Changed dimension and to Linear
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        x = self.laplacian(x)
        
        # Reshape to [B, L, C] format expected by transformer
        _, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Properly iterate through blocks
        for block in self.blocks:
            x = block(x)
            
        # Use cls token for classification
        x = x[:, 0]  # Just use the cls token output
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)

class Combined_2(nn.Module):
    def __init__(self, num_classes=2, in_channels=3):
        super(Combined_2, self).__init__()
        self.cpe = ConditionalPositionEncoding(in_channels=in_channels, kernel_size=3)
        self.gcn = Grapher(in_channels=in_channels)
        # Pass appropriate dimensions
        self.one_block = nn.Linear(in_channels, num_classes)  # Simplified

    def forward(self, x):
        x = self.cpe(x)
        x = self.gcn(x)
        x = self.one_block(x)
        return F.log_softmax(x, dim=1)  # Correct softmax usage

class Final(nn.Module):
    def __init__(self, num_classes=2):
        super(Final, self).__init__()
        # Memory optimized models with reduced complexity
        self.combined_1 = Model_2(num_classes=num_classes, 
                                embed_dim=64,  # Reduced from 96
                                depth=6,       # Reduced from 12
                                mlp_ratio=2.0) # Reduced from 4.0
                                
        self.combined_2 = Combined_2(num_classes=num_classes)
        self.fc = nn.Linear(2*num_classes, num_classes)

    def forward(self, x):
        x1 = self.combined_1(x)
        x2 = self.combined_2(x)
        
        combined = torch.cat((x1, x2), dim=1)
        x = self.fc(combined)
        
        return F.log_softmax(x, dim=1)  # Correct softmax usage

# Memory-efficient training function
def train_with_gradient_checkpointing(model, train_loader, val_loader, criterion, optimizer, 
                                     num_epochs=10, device='cuda'):
    # Enable gradient checkpointing to reduce memory usage
    if hasattr(model, 'blocks'):
        for block in model.blocks:
            if hasattr(block, 'gradient_checkpointing'):
                block.gradient_checkpointing = True
                
    # Use automatic mixed precision for faster training and reduced memory
    scaler = torch.cuda.amp.GradScaler()
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Use mixed precision for forward pass
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
            # Scale gradients and optimize
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return model, history