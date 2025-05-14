import torch
from torch import nn


def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0



class Linear_act_sp(nn.Module):
    def __init__(
        self, 
        in_features, 
        out_features,
        bias=False,
        backend=None
        # device=None,
        # dtype=None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.backend = backend
        self.register_buffer('weight', None)


    def magnitude_pruner(self, x, prune_n=2, prune_m=4, sparsity_ratio=0.5):
        x_metric = torch.abs(x)
        if prune_n != 0:
            x_mask = (torch.zeros_like(x)==1)
            for ii in range(w_metric.shape[1]):
                if ii % prune_m == 0:
                    tmp = x_metric[:,ii:(ii+prune_m)].float()
                    x_mask.scatter_(1, ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
        else:
            thresh = torch.sort(x_metric.flatten().cuda())[0][int(x.numel()*sparsity_ratio)].cpu()
            x_mask = (x_metric<=thresh)

        x_sp = x * (~x_mask)
        return x_sp 


    def forward (self, x):
        bs, seq_len, _ = x.shape
        x_flat = x.view(-1, self.hidden_size)
        
        if self.backend is None:
            out = x @ self.weight.t()
        
        elif self.backend == "2:4_magnitude":
            x_flat_sp = self.magnitude_pruner(x_flat)
            out = x_flat_sp @ self.weight.t()

        return out

    @classmethod
    def from_original(cls, orig_linear, backend=None):
        linear_sp = cls(
            orig_linear.in_features, 
            orig_linear.out_features, 
            backend
        )
        linear_sp = orig_linear.weight

        return linear_sp

