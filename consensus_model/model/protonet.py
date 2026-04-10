import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2) # N x M -> each row represents distances of a point to M proto

class Protonet(nn.Module):
    def __init__(self, encoder, n_ways, n_supports, n_queries, hidden_dims, z_dims):
        super().__init__()
        self.encoder = encoder
        self.n_ways = n_ways
        self.n_supports = n_supports
        self.n_queries = n_queries
        self.hidden_dims = hidden_dims
        self.z_dims = z_dims

    def loss(self, batch, device):
        # classes = batch['precursor']
        n_fragment_types = batch['xs'].shape[-2]
        max_fragment_length = batch['xs'].shape[-1]

        xs = batch['xs']
        xq = batch['xq']
        xs = xs.to(device)
        xq = xq.to(device)
        

        target_inds = torch.arange(0, self.n_ways).view(self.n_ways, 1, 1).expand(self.n_ways, self.n_queries, 1).long().to(device)

        x = torch.cat((xs, xq), dim = 0)
        z = self.encoder(x.view(self.n_ways * (self.n_supports + self.n_queries), 1, n_fragment_types, max_fragment_length)).squeeze() # (B * (S + Q), 1, n_fragment_types, n_fragment_length) -> (B * (S + Q), Z_DIM)
        z_proto = z[:self.n_ways*self.n_supports].view(self.n_ways, self.n_supports, self.z_dims).mean(1) # (B, S, Z_DIM) -> mean() -> (B, Z_DIM)
        zq = z[self.n_ways*self.n_supports:] # (B * Q, Z_DIM)

        dists = euclidean_dist(zq, z_proto) # (W * Q, W)

        log_p_y = F.log_softmax(-dists, dim=1).view(self.n_ways, self.n_queries, -1)
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
        
        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item(),
        }