import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch_geometric.nn import GPSConv , global_mean_pool , GCNConv , GATConv , SAGEConv
from functools import partial
from DeepHypergraph.SHSL.src import hypergraph_learner
from sklearn.cluster import KMeans
import numpy as np
from DeepHypergraph import Hypergraph
from GraphGPS.graphgps.encoder.kernel_pos_encoder import RWSENodeEncoder
from hyp_model import hypergraph_to_edge_index

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

    
def cal_similarity_graph(node_embeddings):
    normalized_embeddings = node_embeddings / torch.norm(node_embeddings, dim=1, keepdim=True)
    similarity_graph = torch.mm(normalized_embeddings, normalized_embeddings.t())
    return similarity_graph  #HW*HW

class att_hgraph_learner(nn.Module):
    def __init__(self, features, fea_dim, num_clusters, num_layers , hidden_channels , out_channels , dim_emb ,  args):
        super(att_hgraph_learner, self).__init__()
        self.layers = nn.ModuleList()
        self.features = features
        self.feat_dim = fea_dim 
        self.num_cluster = num_clusters
        self.args = args 
        self.dim_emb = dim_emb
        
        self.node_enc = RWSENodeEncoder(dim_emb= dim_emb , expand_x= True)
        
        self.GPS = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GPSConv(
                    channels=hidden_channels,  # Use hidden_channels here
                    heads=3,
                    conv=GCNConv(hidden_channels, hidden_channels),  # Instantiate GCNConv directly
                    act='relu',
                    norm='layernorm',
                    attn_type='multihead',
                    dropout=0.1
                )
            )

        self.lin1 = nn.Linear(hidden_channels , hidden_channels)
        self.lin2 = nn.Linear(hidden_channels , out_channels)

    def forward(self , x ):
        B , C , H , W = x.shape 

        num_nodes = H*W
        node_emb = self.node_enc(x)

        print("Node enc" , node_emb)
        similarity = cal_similarity_graph(node_emb) 
        node_features = node_emb.flatten(2).transpose(1 ,2)
        H = Hypergraph.from_feature_kNN(similarity.detach().cpu(), self.num_clusters)

        outputs = []
        for b in range(B):
            edge_index = hypergraph_to_edge_index(similarity , num_nodes )
        for conv in self.convs:
            x_conv = conv(x_conv ,edge_index )

        if x_conv.dim() > 1 and x_conv.size(0) > 1:
            x_conv = global_mean_pool(x_conv , torch.zeros(x_conv.size(0) , dtype = torch.long , device = x_conv.device))
        else:
            x_conv = x_conv.unsqueeze(0)

        outputs.append(x_conv)

        x = torch.cat(outputs , dim = 0)

        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return x
    


            



