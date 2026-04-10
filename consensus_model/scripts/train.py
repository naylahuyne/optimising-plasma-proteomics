import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from enum import Enum
from tqdm import tqdm

from model.protonet import Protonet
from model.encoder import ConvolutionEncoder
from data.data import ProteomeToolsTrainingDataset, ProteomeToolsEvaluationDataset
from data.data import custom_collate
from .utils import get_argparser

from .eval import eval_validation

class EncoderEnum(Enum):
    CNN2D = 0
    CNN1D = 1
    MLP = 2

def train():
    parser = get_argparser()
    args = parser.parse_args()

    data_dir = args.datadir
    n_ways = args.ways
    n_supports = args.supports
    n_queries = args.queries
    learning_rate = args.learning_rate
    n_epoch = args.epoch

    hidden_dims = 64
    z_dims = 32
    
    lookup_df = pd.read_csv(os.path.join(data_dir, 'lookup.tsv'), sep='\t')
    lookup_df['ref'] = lookup_df['Reference table'].apply(lambda x : int(x.split('.')[0]))
    msms_dfs = []
    num_msms_files = lookup_df['ref'].max()
    for i in range(num_msms_files + 1):
        msms_dfs.append(pd.read_csv(os.path.join(data_dir, f"msms_{i:04d}.tsv"), sep='\t'))

    torch.manual_seed(1337)
    train_df = lookup_df.query("`Number of MS/MS` >= 10")
    testval_df = lookup_df.query("`Number of MS/MS` < 10 & `Number of MS/MS` > 5")
    test_df = testval_df.sample(frac=0.5)
    val_df = testval_df.drop(test_df.index)

    encoder = ...
    if args.encoder_id == EncoderEnum.CNN2D.value:
        encoder = ConvolutionEncoder(hidden_dims, z_dims)
    elif args.encoder_id == EncoderEnum.CNN1D.value:
        pass
    elif args.encoder_id == EncoderEnum.MLP.value:
        pass

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(1337)
    model = Protonet(encoder, n_ways, n_supports, n_queries, hidden_dims, z_dims)
    model.to(device)
    train_dataset = ProteomeToolsTrainingDataset(train_df, msms_dfs, n_supports, n_queries)
    val_dataset = ProteomeToolsEvaluationDataset(val_df, msms_dfs, n_supports)
    train_loader = DataLoader(train_dataset, batch_size=n_ways, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=n_ways, shuffle=True, drop_last=True, collate_fn=custom_collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    acc_validation_list = []
    for ep in tqdm(range(n_epoch), desc='Training epoch: '):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            loss_val, metric_dicts = model.loss(batch, device)
            loss_val.backward()
            optimizer.step()
        model.eval()
        acc_validation_list.append(eval_validation(val_loader, encoder, device))
    
    with open('validation_accuracies.txt', 'w') as f:
        for item in acc_validation_list:
            f.write(f"{item}\n")

    torch.save(encoder.state_dict(), "encoder_weights.pth")

train()
        