import torch
from torch.utils.data import Dataset

class ProteomeToolsTrainingDataset(Dataset):
    def __init__(self, lookup_df, msms_dfs, n_supports = 5, n_queries = 5):
        self.lookup_df = lookup_df
        self.n_supports = n_supports
        self.n_queries = n_queries
        self.msms_dfs = msms_dfs

    def __len__(self):
        return self.lookup_df.shape[0]
 
    def __getitem__(self, class_idx):
        precursor_row = self.lookup_df.iloc[class_idx]

        ref = precursor_row['ref']
        start_id = precursor_row['Start id']
        total_examples = precursor_row['Number of MS/MS']

        end_id = start_id + total_examples
        perm = torch.randperm(total_examples)

        supports_ids = perm[:self.n_supports] + start_id
        queries_ids = perm[self.n_supports:self.n_supports + self.n_queries] + start_id

        idx = (self.msms_dfs[ref]['id'] >= start_id) & (self.msms_dfs[ref]['id'] < end_id)
        query = self.msms_dfs[ref].loc[idx]

        xs = []
        xq = []
        for row in query.itertuples():
            if row[1] in supports_ids:
                xs.append(torch.tensor(row[2:]).reshape(4, 39))
            elif row[1] in queries_ids:
                xq.append(torch.tensor(row[2:]).reshape(4, 39))
        return {
            'precursor': self.lookup_df.iloc[class_idx]['Precursor'],
            'xs' : torch.stack(xs),
            'xq' : torch.stack(xq),
        }

class ProteomeToolsEvaluationDataset(Dataset):
    def __init__(self, lookup_df, msms_dfs, n_supports=5, n_queries=0):
        """
        Arguments:
          lookup_file (str): path to the lookup file
          data_dir (str): path to the data directory
        """
        self.lookup_df = lookup_df
        self.msms_dfs = msms_dfs
        self.n_supports = n_supports
        self.n_queries = n_queries

    def __len__(self):
        return self.lookup_df.shape[0]

    def __getitem__(self, class_idx):
        precursor_row = self.lookup_df.iloc[class_idx]
        ref = precursor_row['ref']
        start_id = precursor_row['Start id']
        total_examples = precursor_row['Number of MS/MS']
        n_queries = total_examples - self.n_supports if self.n_queries == 0 else self.n_queries
        end_id = start_id + total_examples
        perm = torch.randperm(total_examples)
        supports_ids = perm[:self.n_supports] + start_id
        queries_ids = perm[self.n_supports:self.n_supports + n_queries] + start_id
        idx = (self.msms_dfs[ref]['id'] >= start_id) & (self.msms_dfs[ref]['id'] < end_id)
        query = self.msms_dfs[ref].loc[idx]
        xs = []
        xq = []
        for row in query.itertuples():
          if row[1] in supports_ids:
            xs.append(torch.tensor(row[2:]).reshape(4, 39))
          elif row[1] in queries_ids:
            xq.append(torch.tensor(row[2:]).reshape(4, 39))
        return {
            'precursor': self.lookup_df.iloc[class_idx]['Precursor'],
            'xs' : torch.stack(xs),
            'xq' : torch.stack(xq),
        }
    
def custom_collate(batch):
    precursor = []
    xs = []
    xq = []
    target_inds = []
    i = 0
    for item in batch:
        precursor.append(item['precursor'])
        xs.append(item['xs'])
        xq.append(item['xq'])
        for _ in range(item['xq'].shape[0]):
            target_inds.append(torch.tensor([i]))
        i += 1
    return {
        'precursor': precursor,
        'xs' : torch.cat(xs, dim=0),
        'xq' : torch.cat(xq, dim=0),
        'target_inds' : torch.cat(target_inds)
    }

class ProteomeToolsPredictionDataset(Dataset):
    def __init__(self, lookup_df, msms_dfs):
        """
        Arguments:
            lookup_file (str): path to the lookup file
            data_dir (str): path to the data directory
        """
        self.lookup_df = lookup_df
        self.msms_dfs = msms_dfs

    def __len__(self):
        return self.lookup_df.shape[0]

    def __getitem__(self, class_idx):
        precursor_row = self.lookup_df.iloc[class_idx]
        ref = precursor_row['ref']

        start_id = precursor_row['Start id']
        total_examples = precursor_row['Number of MS/MS']
        end_id = start_id + total_examples

        idx = (self.msms_dfs[ref]['id'] >= start_id) & (self.msms_dfs[ref]['id'] < end_id)
        query = self.msms_dfs[ref].loc[idx]
        xs = []
        for row in query.itertuples():
            xs.append(torch.tensor(row[2:]).reshape(4, 39))
        return {
            'precursor': self.lookup_df.iloc[class_idx]['Precursor'],
            'ref' : ref,
            'xs' : torch.stack(xs),
            'id' : torch.arange(start_id, end_id),
        }

    