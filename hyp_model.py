import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from hypergraph import ImageToHypergraph
from torch_geometric.nn import GPSConv , global_mean_pool , GCNConv
from ViHGNN.assets.gcn_lib.torch_vertex import HypergraphConv2d


def hypergraph_to_edge_index(hyperedge_matrix, num_points):
    """
    Convert a hypergraph representation to an edge index for GPSConv.
    
    Args:
        hyperedge_matrix: Tensor of shape (batch_size, n_clusters, num_points_index)
        num_points: Total number of points in the graph
        
    Returns:
        edge_index: Tensor of shape [2, num_edges] for GPSConv
    """
    batch_size, n_clusters, _ = hyperedge_matrix.shape
    edge_list = []
    
    for b in range(batch_size):
        for h_idx in range(n_clusters):
            points = hyperedge_matrix[b, h_idx]
            valid_points = points[points >= 0]
            
            if len(valid_points) <= 1:
                continue
            for i in range(len(valid_points)):
                for j in range(i+1, len(valid_points)):
                    edge_list.append([valid_points[i].item(), valid_points[j].item()])
    
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t() 
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    return edge_index


class hyp_model_1(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=7, num_clusters=10, threshold=0.7, m=1):
        super(hyp_model_1, self).__init__()

        self.hgcn = HypergraphConv2d(in_channels, out_channels , act = 'relu' , norm = None , bias = True)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GPSConv(
                    channels=hidden_channels,  # Use hidden_channels here
                    heads=5,
                    conv=GCNConv(hidden_channels, hidden_channels),  # Instantiate GCNConv directly
                    act='relu',
                    norm='layernorm',
                    attn_type='multihead',
                    dropout=0.1
                )
            )

        self.hypergraph_model = ImageToHypergraph(in_chans=in_channels, embed_dim=hidden_channels, 
                                         num_clusters=num_clusters, threshold=threshold, m=m)
        
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x):  # Removed batch_size argument
        batch_size, channels, h, w = x.shape

        hyperedge_matrix, point_hyperedge_index, hyperedge_features, patch_positions, (h, w), x_embed , centers = self.hypergraph_model(x)

        print(hyperedge_matrix.shape)
        print(hyperedge_matrix)

        x_conv_1 = self.hgcn(x , hyperedge_matrix , point_hyperedge_index , centers )

        node_features = x_embed.flatten(2).transpose(1, 2)
        num_points = h * w

        outputs = []
        for b in range(batch_size): 
            edge_index = hypergraph_to_edge_index(hyperedge_matrix[b:b+1], num_points)
            
            features = node_features[b]

            x_conv = features  # Initialize x_conv with features
            for conv in self.convs:
                x_conv = conv(x_conv_1, edge_index)
                
            if x_conv.dim() > 1 and x_conv.size(0) > 1:
                x_conv = global_mean_pool(x_conv, torch.zeros(x_conv.size(0), dtype=torch.long, device=x_conv.device))  # Provide batch tensor
            else:
                x_conv = x_conv.unsqueeze(0)  # Ensure correct dimensions for concatenation
            outputs.append(x_conv)


        x = torch.cat(outputs, dim=0)

        # Apply final layers
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return x

    

        
        
        
        
        