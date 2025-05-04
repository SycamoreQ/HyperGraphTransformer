import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import hsv_to_rgb
import numpy as np
from torch_geometric.transforms import AddRandomWalkPE
from encoder import PatchEmbed

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


def initialize_memberships(batch_size, n_points, n_clusters, device):
    """
    Initialize the membership matrix for Fuzzy C-Means clustering.

    Args:
        batch_size: int
        n_points: int
        n_clusters: int
        device: torch.device

    Returns:
        memberships: tensor (batch_size, n_points, n_clusters)
    """
    memberships = torch.rand(batch_size, n_points, n_clusters, device=device)
    memberships = memberships / memberships.sum(dim=2, keepdim=True)
    return memberships


def fuzzy_c_means(x, n_clusters, m=2, epsilon=1e-6, max_iter=1000):
    """
    Fuzzy C-Means clustering

    Args:
        x: tensor (batch_size, num_dims, num_points, 1)
        n_clusters: int, the number of clusters
        m: float, fuzziness parameter
        epsilon: float, threshold for stopping criterion
        max_iter: int, maximum number of iterations

    Returns:
        membership: tensor (batch_size, num_points, n_clusters)
        centers: tensor (batch_size, num_dims, n_clusters)
    """
    batch_size, num_dims, num_points, _ = x.size()
    x = x.squeeze(-1).transpose(1, 2)  # Shape: (batch_size, num_points, num_dims)

    memberships = initialize_memberships(batch_size, num_points, n_clusters, x.device)

    centers = torch.zeros(batch_size, num_dims, n_clusters, device=x.device)
    prev_memberships = torch.zeros_like(memberships)

    for iteration in range(max_iter):
        for cluster in range(n_clusters):
            weights = memberships[:, :, cluster] ** m
            denominator = weights.sum(dim=1, keepdim=True)
            numerator = (weights.unsqueeze(2) * x).sum(dim=1)
            centers[:, :, cluster] = numerator / denominator

        for cluster in range(n_clusters):
            diff = x - centers[:, :, cluster].unsqueeze(1)
            dist = torch.norm(diff, p=2, dim=2) 
            memberships[:, :, cluster] = 1.0 / (dist ** (2 / (m - 1)))

        memberships_sum = memberships.sum(dim=2, keepdim=True)
        memberships = memberships / memberships_sum
        if iteration > 0 and torch.norm(prev_memberships - memberships) < epsilon:
            break
        prev_memberships = memberships.clone()

    return memberships, centers


def construct_hyperedges(x, num_clusters, threshold=0.5, m=2):
    """
    Constructs hyperedges based on fuzzy c-means clustering.

    Args:
        x (torch.Tensor): Input point cloud data with shape (batch_size, num_dims, num_points, 1).
        num_clusters (int): Number of clusters (hyperedges).
        threshold (float): Threshold value for memberships to consider a point belonging to a cluster.
        m (float): Fuzzifier for fuzzy c-means clustering.

    Returns:
        hyperedge_matrix (torch.Tensor): Tensor of shape (batch_size, n_clusters, num_points_index).
            Represents each cluster's points. Padded with -1 for unequal cluster sizes.
        point_hyperedge_index (torch.Tensor): Tensor of shape (batch_size, num_points, cluster_index).
            Indicates the clusters each point belongs to. Padded with -1 for points belonging to different numbers of clusters.
        hyperedge_features (torch.Tensor): Tensor of shape (batch_size, num_dims, n_clusters).
            The center of each cluster, serving as the feature for each hyperedge.
    """
    
    with torch.no_grad():
        x = x.detach()  # Detach x from the computation graph
        
        batch_size, num_dims, num_points, _ = x.shape
        
        memberships, centers = fuzzy_c_means(x, num_clusters, m)
        hyperedge_matrix = -torch.ones(batch_size, num_clusters, num_points, dtype=torch.long, device=x.device)
        for b in range(batch_size):
            for c in range(num_clusters):
                idxs = torch.where(memberships[b, :, c] > threshold)[0]
                hyperedge_matrix[b, c, :len(idxs)] = idxs
        
        max_edges_per_point = (memberships > threshold).sum(dim=-1).max().item()
        point_hyperedge_index = -torch.ones(batch_size, num_points, max_edges_per_point, dtype=torch.long, device=x.device)
        for b in range(batch_size):
            for p in range(num_points):
                idxs = torch.where(memberships[b, p, :] > threshold)[0]
                point_hyperedge_index[b, p, :len(idxs)] = idxs
    
    return hyperedge_matrix, point_hyperedge_index, centers


class ImageToHypergraph(nn.Module):
    """
    Model that takes an image, applies patch embedding, and constructs a hypergraph.
    """
    def __init__(self, in_chans=3, embed_dim=96, num_clusters=10, threshold=0.5, m=2):
        """
        Args:
            in_chans: Number of input channels.
            embed_dim: Dimension of patch embeddings.
            num_clusters: Number of clusters (hyperedges).
            threshold: Threshold for cluster membership.
            m: Fuzzifier parameter for fuzzy c-means.
        """
        super().__init__()
        self.RWSE = AddRandomWalkPE(walk_length = 10)
        self.PatchEmbed()
        self.num_clusters = num_clusters
        self.threshold = threshold
        self.m = m
        
    def forward(self, x):
        """
        Args:
            x: Image tensor of shape (B, C, H, W)
            
        Returns:
            hyperedge_matrix: Tensor representing hyperedges
            point_hyperedge_index: Tensor mapping points to hyperedges
            hyperedge_features: Features of hyperedges (cluster centers)
            patch_positions: Position of each patch in the original image
            x_embed: Patch Embedding
        """
        x_embed_1 = self.PE(x)
        x_embed = self.RWSE(x)
        
        batch_size, channels, h, w = x_embed.shape
        y_positions = torch.arange(0, h, device=x.device)
        x_positions = torch.arange(0, w, device=x.device)
        y_grid, x_grid = torch.meshgrid(y_positions, x_positions, indexing='ij')
        patch_positions = torch.stack([y_grid.flatten(), x_grid.flatten()], dim=1)
        
        x_reshape = x_embed.flatten(2).unsqueeze(-1)  # (B, C, H*W, 1)
        
        # Construct hypergraph
        hyperedge_matrix, point_hyperedge_index, hyperedge_features = construct_hyperedges(
            x_reshape, self.num_clusters, self.threshold, self.m
        )

        memberships, centers = fuzzy_c_means(x, num_clusters = 10 , m = 2)
        
        return hyperedge_matrix, point_hyperedge_index, hyperedge_features, patch_positions, (h, w) , x_embed , centers


def visualize_hypergraph_patches(image_path, hyperedge_matrix, patch_positions, h, w, orig_size=(224, 224), output_path=None):
    """
    Visualize the hypergraph structure on the original image.
    
    Args:
        image_path: Path to the original image.
        hyperedge_matrix: Matrix representing which patches belong to which hyperedges.
        patch_positions: Positions of patches.
        h, w: Height and width of the feature map after patch embedding.
        orig_size: Original size the image was resized to before processing.
        output_path: Path to save the visualization. If None, display the image.
    """
    # Load original image
    image = Image.open(image_path).convert('RGB')
    image = image.resize(orig_size)
    
    # Calculate patch size in original image coordinates
    patch_size_y = orig_size[0] / h
    patch_size_x = orig_size[1] / w
    
    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)
    
    # Generate colors for each hyperedge
    num_hyperedges = hyperedge_matrix.shape[1]
    colors = [hsv_to_rgb((i/num_hyperedges, 0.8, 0.8)) for i in range(num_hyperedges)]
    
    # Plot patches for each hyperedge
    for h_idx in range(num_hyperedges):
        # Get indices of patches in this hyperedge
        patch_indices = hyperedge_matrix[0, h_idx]
        patch_indices = patch_indices[patch_indices >= 0]  # Remove padding (-1)
        
        if len(patch_indices) == 0:
            continue
        
        # Plot each patch
        for idx in patch_indices:
            y, x = patch_positions[idx]
            y_coord = y * patch_size_y
            x_coord = x * patch_size_x
            
            rect = mpatches.Rectangle(
                (x_coord, y_coord), patch_size_x, patch_size_y, 
                linewidth=1, edgecolor=colors[h_idx], facecolor='none', alpha=0.7
            )
            ax.add_patch(rect)
    
    # Add legend
    legend_patches = [mpatches.Patch(color=colors[i], label=f'Hyperedge {i+1}') 
                     for i in range(num_hyperedges)]
    ax.legend(handles=legend_patches, loc='best')
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Save or display
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    else:
        plt.tight_layout()
        plt.show()
    
    plt.close()






        
    





        
        
        
        
        

        
        
        
        

