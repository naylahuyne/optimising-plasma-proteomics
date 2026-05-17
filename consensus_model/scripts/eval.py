import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from model.protonet import euclidean_dist, Protonet
from model.encoder import TransformerEncoderConfig, TransformerEncoder
from data.data import ProteomeToolsEvaluationDataset, custom_collate

import pandas as pd

import os
import argparse

def eval_validation(eval_loader, encoder, device):
    mean_acc = 0
    n_batch = 0
    for batch in eval_loader:
        n_fragment_types = batch['xs'].shape[-2]
        max_fragment_length = batch['xs'].shape[-1]
        xs = batch['xs'].to(device)
        xq = batch['xq'].to(device)
        n_classes = len(batch['precursor'])
        n_supports = xs.shape[0] // n_classes
        target_inds = batch['target_inds'].to(device)
        

        x = torch.cat((xs, xq), dim = 0)
        z = encoder(x.view(-1, n_fragment_types, max_fragment_length)).squeeze() # (B = W*(S+Q), Ty, L) -> (B, Z)
        z_dim = z.shape[-1]
        z_proto = z[:n_classes*n_supports].view(n_classes, n_supports, z_dim).mean(1) # (W, S, Z) -> mean(1) -> (W, Z)
        zq = z[n_classes*n_supports:] # (nQ, Z)

        dists = euclidean_dist(zq, z_proto) # (nQ, W)

        log_p_y = F.log_softmax(-dists, dim=1)
        _, y_hat = log_p_y.max(1)
        acc_val = torch.eq(y_hat.squeeze(), target_inds).float().mean()
        n_batch += 1
        mean_acc += acc_val
    return mean_acc / n_batch

def get_argparser():
    parser = argparse.ArgumentParser(description="Predict")
    parser.add_argument("--datadir",
                        type=str,
                        help="path to data directory",
                        required=True
                        )
    parser.add_argument("--model_path",
                        type=str,
                        help="path to state dict of encoder model",
                        required=True    
                        )
    parser.add_argument("-w", "--ways", 
                        type=int,
                        help="number of classes to classify into",
                        required=True
                        )
    parser.add_argument("-s", "--supports",
                        type=int,
                        help="number of examples per class",
                        required=True
                        )
    return parser

if __name__ == "__main__":
    arg_parser = get_argparser()
    args = arg_parser.parse_args()
    data_dir = args.datadir
    n_ways = args.ways
    n_supports = args.supports

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = TransformerEncoderConfig(39, 6, 8, 64, 32, 0.1, device=device)
    encoder = TransformerEncoder(config)
    encoder = encoder.to(device)
    encoder.load_state_dict(torch.load(args.model_path, map_location=torch.device(device)))
    encoder.eval()

    lookup_df = pd.read_csv(os.path.join(data_dir, 'lookup.tsv'), sep='\t')
    lookup_df['ref'] = lookup_df['Reference table'].apply(lambda x : int(x.split('.')[0]))
    msms_dfs = []
    num_msms_files = lookup_df['ref'].max()
    for i in range(num_msms_files + 1):
        msms_dfs.append(pd.read_csv(os.path.join(data_dir, f"msms_{i:04d}.tsv"), sep='\t'))

    testval_df = lookup_df.query("`Number of MS/MS` < 10 & `Number of MS/MS` > 5")
    test_df = testval_df.sample(frac=0.5, random_state=1337)

    torch.manual_seed(1337)
    test_dataset = ProteomeToolsEvaluationDataset(test_df, msms_dfs, n_supports)
    test_loader = DataLoader(test_dataset, batch_size=n_ways, shuffle=True, drop_last=True, collate_fn=custom_collate)

    with open('test_accuracies.txt', 'w') as f:
        f.write(f"{eval_validation(test_loader, encoder, device)}\n")