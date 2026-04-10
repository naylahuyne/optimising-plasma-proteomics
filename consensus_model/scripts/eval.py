import torch
from model.protonet import euclidean_dist
from torch.nn import functional as F

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
        z = encoder(x.view(-1, 1, n_fragment_types, max_fragment_length)).squeeze() # (B * (S + Q), 1, n_fragment_types, n_fragment_length) -> (B * (S + Q), Z_DIM)
        z_dim = z.shape[-1]
        z_proto = z[:n_classes*n_supports].view(n_classes, n_supports, z_dim).mean(1) # (B, S, Z_DIM) -> mean() -> (B, Z_DIM)
        zq = z[n_classes*n_supports:] # (B * Q, Z_DIM)

        dists = euclidean_dist(zq, z_proto) # (W * Q, W)

        log_p_y = F.log_softmax(-dists, dim=1)
        _, y_hat = log_p_y.max(1)
        acc_val = torch.eq(y_hat.squeeze(), target_inds).float().mean()
        n_batch += 1
        mean_acc += acc_val
    return mean_acc / n_batch