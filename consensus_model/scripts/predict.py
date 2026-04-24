import torch
from data.data import ProteomeToolsPredictionDataset
from model.encoder import TransformerEncoder, TransformerEncoderConfig

import pandas as pd

import os

from tqdm import tqdm

def euclidean_dist(x, y):
    # x: N x D
    # y: D
    n = x.size(0)
    d = x.size(1)
    assert d == y.size(0)

    y = y.unsqueeze(0).expand(n, d)

    return torch.pow(x - y, 2).sum(1) # N x M -> each row represents distances of a point to M proto

def predict():
    choices = []
    data_dir = "..\\data-collection\\final_data"
    lookup_df = pd.read_csv(os.path.join(data_dir, 'lookup.tsv'), sep='\t')[:1000]
    lookup_df['ref'] = lookup_df['Reference table'].apply(lambda x : int(x.split('.')[0]))
    msms_dfs = []
    num_msms_files = lookup_df['ref'].max()
    for i in range(num_msms_files + 1):
        msms_dfs.append(pd.read_csv(os.path.join(data_dir, f"msms_{i:04d}.tsv"), sep='\t'))
    dataset = ProteomeToolsPredictionDataset(lookup_df, msms_dfs)
    config = TransformerEncoderConfig(39, 6, 8, 64, 32, 0.1)
    encoder = TransformerEncoder(config)
    encoder.load_state_dict(torch.load("transformer_encoder_weights.pth", map_location=torch.device('cpu')))
    encoder.eval()
    for precursor in tqdm(dataset):
        xs = precursor['xs']
        ids = precursor['id']
        ref = precursor['ref']
        z = encoder(xs) # S x Z
        z_proto = z.mean(0) # Z
        dist = euclidean_dist(z, z_proto)
        print(dist)
        idx = torch.argmin(dist)
        choices.append((ref, ids[idx]))
        
    
    with open('choices.txt', 'w') as file:
        for c in choices:
            file.write(str(c[0].item()))
            file.write(" ")
            file.write(str(c[1].item()))
            file.write("\n")

predict()