import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange, repeat
import numpy as np

class LaplacianPositionalEncoding(nn.Module):
    def __init__(self, patch_size, img_size, dim, normalized=True):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.dim = dim
        self.normalized = normalized
        
        # Calculate grid size
        self.grid_h = img_size // patch_size
        self.grid_w = img_size // patch_size
        self.num_patches = self.grid_h * self.grid_w
        
        # Initialize projection layer properly
        self.projection = nn.Linear(min(self.num_patches, dim), dim)
        
        # Compute Laplacian (moved to forward to handle dynamic sizes)
        self.register_buffer('edge_index', self.create_edge_index())
        
        # Create static Laplacian matrix to avoid recalculation
        adj = self.create_adjacency_matrix()
        degree = torch.diag(adj.sum(dim=1))
        laplacian = degree - adj
        
        # Pre-compute eigenvectors for efficiency
        eigvals, eigvecs = torch.linalg.eigh(laplacian)
        k = min(self.num_patches, self.dim)
        basis = eigvecs[:, :k]  # [N, k]
        
        if self.normalized:
            weights = 1.0 / torch.sqrt(eigvals[:k] + 1e-6)
            weights[0] = 0  
            basis = basis * weights.unsqueeze(0)
        
        pos_emb = self.projection(basis)  # [N, dim]
        
        self.register_buffer('pos_emb', pos_emb)
        
    def create_edge_index(self):
        """Create edge connections for the grid"""
        edge_index = []
        for i in range(self.grid_h):
            for j in range(self.grid_w):
                node = i * self.grid_w + j
                if i > 0: edge_index.append([node, (i-1)*self.grid_w + j])
                if j > 0: edge_index.append([node, i*self.grid_w + (j-1)])
                if i < self.grid_h - 1: edge_index.append([node, (i+1)*self.grid_w + j])
                if j < self.grid_w - 1: edge_index.append([node, i*self.grid_w + (j+1)])
        return torch.tensor(edge_index).t()
    
    def create_adjacency_matrix(self):
        """Create adjacency matrix from edge index"""
        adj = torch.zeros(self.num_patches, self.num_patches)
        for i in range(self.edge_index.size(1)):
            src, dst = self.edge_index[0, i], self.edge_index[1, i]
            adj[src, dst] = 1
            adj[dst, src] = 1  
        return adj
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, N, C] where N is num_patches
                or [B, C, H, W] for image input
        Returns:
            x: Input with positional encoding added
        """
        if len(x.shape) == 4:
            B, C, H, W = x.shape
            assert H == self.grid_h and W == self.grid_w, \
                f"Input size {H}x{W} doesn't match expected {self.grid_h}x{self.grid_w}"
            x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        else:
            B, N, C = x.shape
            assert N == self.num_patches, f"Expected {self.num_patches} patches, got {N}"
        
        pos_emb = self.pos_emb.unsqueeze(0).expand(B, -1, -1)  # [B, N, dim]
        
        if C != self.dim:
            if C < self.dim:
                pos_emb = pos_emb[:, :, :C]
            else: 
                pos_emb = F.pad(pos_emb, (0, C - self.dim))
        
        x = x + pos_emb
        
        return x

class PatchEmbed(nn.Module):
    """
    Patch embedding block based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """

    def __init__(self, in_chans=3, in_dim=64, dim=96):
        """
        Args:
            in_chans: number of input channels.
            dim: feature size dimension.
        """
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU()
            )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x

    

class ConditionalPositionEncoding(nn.Module):
    """
    Implementation of conditional positional encoding. For more details refer to paper: 
    `Conditional Positional Encodings for Vision Transformers <https://arxiv.org/pdf/2102.10882.pdf>`_
    """
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.pe = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=in_channels
        )

    def forward(self, x):
        x = self.pe(x) + x
        return x

# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
# modified from https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------



# --------------------------------------------------------
# relative position embedding
# References: https://arxiv.org/abs/2009.13658
# --------------------------------------------------------
def get_2d_relative_pos_embed(embed_dim, grid_size):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, grid_size*grid_size]
    """
    pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size)
    relative_pos = 2 * np.matmul(pos_embed, pos_embed.transpose()) / pos_embed.shape[1]
    return relative_pos


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb