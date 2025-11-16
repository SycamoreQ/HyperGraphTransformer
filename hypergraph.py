import torch
import torch.nn.functional as F

@torch.no_grad()
def _init_memberships(n_points, n_clusters, device):
    u = torch.rand(n_points, n_clusters, device=device)
    u = u / (u.sum(dim=1, keepdim=True) + 1e-12)
    return u

@torch.no_grad()
def fuzzy_c_means_single(x, n_clusters, m=1.5, eps=1e-4, max_iter=100):
    N, D = x.shape
    device = x.device
    u = _init_memberships(N, n_clusters, device)               
    for _ in range(max_iter):
        um = u.pow(m)                                          
        denom = um.sum(dim=0, keepdim=True).clamp_min(1e-12)    
        centers = (um.t() @ x) / denom.t()                      
        # distances: [N, C]
        dist = torch.cdist(x, centers, p=2).clamp_min(1e-8)
        inv = dist.pow(-2.0/(m-1.0))
        u_next = inv / (inv.sum(dim=1, keepdim=True) + 1e-12)
        if torch.max(torch.abs(u_next - u)) < eps:
            u = u_next
            break
        u = u_next
    return u, centers                                            # [N, C], [C, D]

@torch.no_grad()
def construct_hyperedges_from_features(x_nodes, num_clusters=8, threshold=0.5, m=1.5):
    u, centers = fuzzy_c_means_single(x_nodes, num_clusters, m=m)
    hyperedges = []
    for c in range(num_clusters):
        idx = torch.nonzero(u[:, c] > threshold, as_tuple=False).flatten()
        if idx.numel() > 0:
            hyperedges.append(idx)
    return hyperedges, u, centers                           

@torch.no_grad()
def hyperedges_to_edge_index(hyperedges, num_nodes):
    edges = set()
    for idx in hyperedges:
        if idx.numel() < 2:
            continue
        nodes = idx.tolist()
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                a, b = nodes[i], nodes[j]
                if 0 <= a < num_nodes and 0 <= b < num_nodes:
                    edges.add((a, b))
                    edges.add((b, a))
    if not edges:
        return torch.empty(2, 0, dtype=torch.long)
    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    return edge_index
