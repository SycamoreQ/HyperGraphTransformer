import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import SAGEConv, global_mean_pool
from encoder import PatchEmbed
from graph_creation import DenseDilatedKnnGraph
from hypergraph import construct_hyperedges_from_features, hyperedges_to_edge_index

def build_knn_edge_index_from_features(x_nodes, k=12, dilation=1):
    # x_nodes: [N, D]
    # DenseDilatedKnnGraph expects [B, D, N, 1]
    B = 1
    N, D = x_nodes.shape
    x4 = x_nodes.t().unsqueeze(0).unsqueeze(-1).contiguous()     # [1, D, N, 1]
    knn = DenseDilatedKnnGraph(k=k, dilation=dilation)
    edge_idx_dense = knn(x4)                                     # [2, B, N, k]
    nn_idx = edge_idx_dense[0, 0]                                # [N, k]
    ctr_idx = edge_idx_dense[1, 0]                               # [N, k] (0..N-1)
    src = ctr_idx.reshape(-1)
    dst = nn_idx.reshape(-1)
    edge_index = torch.stack([src, dst], dim=0)                  # [2, N*k]
    return edge_index

class GNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.conv = SAGEConv(in_ch, out_ch)
        self.norm = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.relu(x, inplace=True)
        x = self.norm(x)
        x = self.drop(x)
        return x

class HyperVisionNet(nn.Module):
    def __init__(
        self,
        num_classes=2,
        in_chans=3,
        patch_embed_dim=96,
        gnn_hidden=128,
        gnn_layers=3,
        k=12,
        dilation=1,
        use_hyperedges=True,
        num_clusters=8,
        hyper_threshold=0.5,
        dropout=0.2,
    ):
        super().__init__()
        self.embed = PatchEmbed(in_chans=in_chans, in_dim=64, dim=patch_embed_dim)
        self.k = k
        self.dilation = dilation
        self.use_hyper = use_hyperedges
        self.num_clusters = num_clusters
        self.hyper_thr = hyper_threshold

        blocks = []
        cin = patch_embed_dim
        for _ in range(gnn_layers):
            blocks.append(GNNBlock(cin, gnn_hidden, dropout=dropout))
            cin = gnn_hidden
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Sequential(
            nn.Linear(gnn_hidden, gnn_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden, num_classes),
        )

    def _image_to_graph(self, img, y=None):
        # img: [3, H, W] -> features [C, h, w] -> nodes [N, C]
    
        device = img.device
        
        feat = self.embed(img.unsqueeze(0)).squeeze(0)           # [C, h, w]
        C, h, w = feat.shape
        x_nodes = feat.permute(1, 2, 0).reshape(h * w, C).contiguous()  # [N, C]
        
        edge_index = build_knn_edge_index_from_features(x_nodes, k=self.k, dilation=self.dilation)
        
        if self.use_hyper:
            # These functions likely return CPU tensors
            hypers, _, _ = construct_hyperedges_from_features(x_nodes, num_clusters=self.num_clusters, threshold=self.hyper_thr)
            hyper_edges = hyperedges_to_edge_index(hypers, num_nodes=x_nodes.size(0))
            
            if hyper_edges.numel() > 0:
                hyper_edges = hyper_edges.to(device)

                edge_index = torch.cat([edge_index, hyper_edges], dim=1)
                edge_index = torch.unique(edge_index.t(), dim=0).t().contiguous()
        
        data = Data(x=x_nodes, edge_index=edge_index, y=y)
        return data

    def forward_graph(self, batch):
        # batch: PyG Batch with .x, .edge_index, .batch
        x, edge_index, b_ix = batch.x, batch.edge_index, batch.batch
        for blk in self.blocks:
            x = blk(x, edge_index)
        g = global_mean_pool(x, b_ix)
        logits = self.head(g)
        return logits

    def forward(self, images, labels=None):
        # images: [B, 3, H, W]
        data_list = []
        for i in range(images.size(0)):
            y = labels[i].view(1) if labels is not None else None
            data = self._image_to_graph(images[i], y=y)
            data_list.append(data)
        batch = Batch.from_data_list(data_list)
        logits = self.forward_graph(batch)
        return logits
