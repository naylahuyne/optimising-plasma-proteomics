import torch
from torch.utils.data import DataLoader

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

def predict():
    dataset = ...
    dataloader = DataLoader(dataset, batch_size=1)
    encoder = ...
    for precursor in dataloader:
        x = precursor['x'] # S x I
        ids = precursor['id']
        z = encoder(x) # S x Z
        z_proto = z.mean(0) # Z
        dist = euclidean_dist(z, z_proto)
        idx = torch.argmax(dist)
        print(ids[id])

