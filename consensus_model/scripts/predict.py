import torch
from torch.utils.data import DataLoader
from data.data import ProteomeToolsPredictionDataset, custom_collate_predict
from model.encoder import TransformerEncoder, TransformerEncoderConfig
import argparse

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

def get_argparser():
    parser = argparse.ArgumentParser(description="Predict")
    parser.add_argument("--datadir",
                        type=str,
                        help="path to data directory for training",
                        required=True
                        )
    parser.add_argument("--precursors_per_run",
                        type=int,
                        help="number of precursors to run in parallel",
                        required=True
                        )
    parser.add_argument("--model_path",
                        type=str,
                        help="path to state dict of encoder model",
                        required=True    
                        )
    return parser

def predict():
    arg_parser = get_argparser()
    args = arg_parser.parse_args()
    choices = []
    data_dir = args.datadir
    lookup_df = pd.read_csv(os.path.join(data_dir, 'lookup.tsv'), sep='\t')
    lookup_df['ref'] = lookup_df['Reference table'].apply(lambda x : int(x.split('.')[0]))
    msms_dfs = []
    num_msms_files = lookup_df['ref'].max()
    for i in range(num_msms_files + 1):
        msms_dfs.append(pd.read_csv(os.path.join(data_dir, f"msms_{i:04d}.tsv"), sep='\t'))
    dataset = ProteomeToolsPredictionDataset(lookup_df, msms_dfs)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = TransformerEncoderConfig(39, 6, 8, 64, 32, 0.1, device=device)
    encoder = TransformerEncoder(config)
    encoder = encoder.to(device)
    encoder.load_state_dict(torch.load(args.model_path, map_location=torch.device(device)))
    encoder.eval()
    dataloader = DataLoader(dataset, batch_size=args.precursors_per_run, collate_fn=custom_collate_predict, shuffle=False, drop_last=False)
    for precursor in tqdm(dataloader):
        xs = precursor['xs'].to(device)
        ids = precursor['id']
        ref = precursor['ref']
        z = encoder(xs) # B x Z, where B = number of spectrum
        start = 0
        for iter in range(len(ids)):
            length = ids[iter].shape[0]
            sub_z = z[start:start+length]
            z_proto = sub_z.mean(0) # B x Z
            dist = euclidean_dist(sub_z, z_proto)
            idx = torch.argmin(dist)
            choices.append((ref[iter], ids[iter][idx]))
            start += length

    with open('choices.txt', 'w') as file:
        for c in choices:
            file.write(str(c[0].item()))
            file.write(" ")
            file.write(str(c[1].item()))
            file.write("\n")

predict()