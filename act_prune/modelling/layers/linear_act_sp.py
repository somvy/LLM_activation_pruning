import torch
from torch import nn


from torch.sparse import (
    SparseSemiStructuredTensorCUTLASS, 
    SparseSemiStructuredTensorCUSPARSELT
)

# from fast_hadamard_transform import hadamard_transform

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
        backend=None,
        name=None
        # device=None,
        # dtype=None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.backend = backend
        self.register_buffer('weight', None)
        self.name = name


    def magnitude_pruner(self, x, prune_n=2, prune_m=4, sparsity_ratio=0.5):
        x_metric = torch.abs(x)

        orig_shape = x.shape
        x_1d = x.view(-1, prune_m)

        
        _, idx = torch.topk(x_1d.abs(), prune_n, dim=1, sorted=False)
        mask_1d = torch.zeros_like(x_1d)
        mask_1d.scatter_(dim=1, index=idx, value=True)
        mask = mask_1d.view(orig_shape)
        x_sp = x * mask
        
        return x_sp 

    def magnitude_pruner_shift(self, x, prune_n=2, prune_m=4, sparsity_ratio=0.5):

        # x_median = torch.median(x, dim=0)[0].reshape(1, -1)
        x_mean = torch.mean(x, dim=0).reshape(1, -1)
        x = x - x_mean
        
        x_metric = torch.abs(x)

        orig_shape = x.shape
        x_1d = x.view(-1, prune_m)
        
        _, idx = torch.topk(x_1d.abs(), prune_n, dim=1, sorted=False)
        mask_1d = torch.zeros_like(x_1d)
        mask_1d.scatter_(dim=1, index=idx, value=True)
        mask = mask_1d.view(orig_shape)
        x_sp = x * mask
        
        return x_sp, x_mean

    def magnitude_pruner_scale(self, x, w, prune_n=2, prune_m=4, sparsity_ratio=0.5):

        x_max = torch.max(x.abs(), dim=0)[0].reshape(1, -1)
        w_max = torch.max(w.abs(), dim=0)[0].reshape(1, -1)
        # alpha_scale = torch.sqrt(x_max) / torch.sqrt(w_max)

        alpha_scale = torch.pow(x_max, 1/2)
        orig_shape = x.shape
        x_scaled = x * (1 / alpha_scale)
        x_1d = x_scaled.view(-1, prune_m)
        
        _, idx = torch.topk(x_1d.abs(), prune_n, dim=1, sorted=False)
        mask_1d = torch.zeros_like(x_1d)
        mask_1d.scatter_(dim=1, index=idx, value=True)
        mask = mask_1d.view(orig_shape)

        x_sp = x_scaled * mask

        return x_sp, alpha_scale

    def magnitude_pruner_hadamard(self, x, prune_n=2, prune_m=4, sparsity_ratio=0.5):      

        orig_shape = x.shape
        x_transformed = hadamard_transform(x, scale=1.0)
        x_1d = x_transformed.view(-1, prune_m)
        _, idx = torch.topk(x_1d.abs(), prune_n, dim=1, sorted=False)
        mask_1d = torch.zeros_like(x_1d)
        mask_1d.scatter_(dim=1, index=idx, value=True)
        mask = mask_1d.view(orig_shape)
        x_sp = x_transformed * mask
        x_sp = hadamard_transform(x_sp, scale=1.0) 

        return x_sp

    def magnitude_var_correction(self, x, w, prune_n=2, prune_m=4, sparsity_ratio=0.5):

        orig_shape = x.shape
        x_1d = x.view(-1, prune_m)
        _, idx = torch.topk(x_1d.abs(), prune_n, dim=1, sorted=False)
        mask_1d = torch.zeros_like(x_1d, dtype=torch.bool)
        mask_1d.scatter_(dim=1, index=idx, value=True)
        mask = mask_1d.view(orig_shape)
        x_sp = x * mask
        x_sp = x_sp * torch.sqrt(x.var(dim=1) / x_sp.var(dim=1)).view(-1, 1)

        return x_sp

    def magnitude_weight_var_correction(self, x, w, prune_n=2, prune_m=4, sparsity_ratio=0.5):

        orig_shape = x.shape
        # x_metric = x.abs() * torch.max(w, dim=0)[0].view(1, -1)
        
        # x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True)
        # x_metric = (x - x_mean).abs() / x_std
        # x_metric = (x - x_mean).abs()

        x_metric = x.abs() * torch.diag(w.t() @ w)
        x_1d = x_metric.view(-1, prune_m)
        _, idx = torch.topk(x_1d, prune_n, dim=1, sorted=False)
        mask_1d = torch.zeros_like(x_1d, dtype=torch.bool)
        mask_1d.scatter_(dim=1, index=idx, value=True)
        mask = mask_1d.view(orig_shape)
        x_sp = x * mask
        # x_bias = (x * (~mask)).mean(dim=0)
        # x_sp += x_bias

        # x_sp = x_sp * torch.sqrt(x.var(dim=0) / x_sp.var(dim=0))
        x_sp = x_sp * torch.sqrt(x.var(dim=1) / x_sp.var(dim=1)).view(-1, 1)
        
        # x_sp = (x_sp + x.mean(dim=1, keepdim=True) - x_sp.mean(dim=1, keepdim=True)) * torch.sqrt(x.var(dim=1) / x_sp.var(dim=1)).view(-1, 1)
        # if self.name in ["o_proj", "down_proj"]:
        #     x_sp = x_sp * torch.sqrt(x.var(dim=1) / x_sp.var(dim=1)).view(-1, 1)

        return x_sp
    
    def magnitude_weight_outliers_var_correction(self, x, w, prune_n=2, prune_m=4, sparsity_ratio=0.5, outlier_tokens=64):

        orig_shape = x.shape
        # x_metric = x.abs() * torch.max(w, dim=0)[0].view(1, -1)

        x_metric = x.abs()
        x_1d = x_metric.view(-1, prune_m)
        _, idx = torch.topk(x_1d, prune_n, dim=1, sorted=False)
        mask_1d = torch.zeros_like(x_1d, dtype=torch.bool)
        mask_1d.scatter_(dim=1, index=idx, value=True)
        mask = mask_1d.view(orig_shape)
        mask[0:outlier_tokens] = True
        mask[-outlier_tokens:] = True
        x_sp = x * mask
        # x_bias = (x * (~mask)).mean(dim=0)
        # x_sp += x_bias
        # x_sp = x_sp * torch.sqrt(x.var(dim=1) / x_sp.var(dim=1)).view(-1, 1)
        x_sp = x_sp * torch.sqrt(x.var(dim=1) / x_sp.var(dim=1)).view(-1, 1)

        return x_sp

    def magnitude_weight_metric_correction(self, x, w, prune_n=2, prune_m=4, sparsity_ratio=0.5):

        orig_shape = x.shape

        x_metric = x.abs() * torch.diag(w.t() @ w).view(1, -1)
        x_1d = x_metric.view(-1, prune_m)
        _, idx = torch.topk(x_1d, prune_n, dim=1, sorted=False)
        mask_1d = torch.zeros_like(x_1d, dtype=torch.bool)
        mask_1d.scatter_(dim=1, index=idx, value=True)
        mask = mask_1d.view(orig_shape)
        x_sp = x * mask
        # x_bias = (x * (~mask)).mean(dim=0)
        # x_sp += x_bias
        # x_sp = x_sp * torch.sqrt(x.var(dim=1) / x_sp.var(dim=1)).view(-1, 1)
        x_sp = x_sp * torch.sqrt(x.var(dim=1) / x_sp.var(dim=1)).view(-1, 1)

        return x_sp

    def forward (self, x):
        bs, seq_len, _ = x.shape
        x_flat = x.view(-1, self.in_features)
        
        if self.backend is None:
            out = x @ self.weight.t()
        
        elif self.backend == "2:4_magnitude":
            x_flat_sp = self.magnitude_pruner(x_flat, prune_n=2, prune_m=4)
            out = x_flat_sp @ self.weight.t()

        elif self.backend == "2:4_magnitude_shift":
            x_flat_sp, x_mean = self.magnitude_pruner_shift(x_flat, prune_n=2, prune_m=4)
            out = x_flat_sp @ self.weight.t() + x_mean @ self.weight.t()
            # out = x_flat_sp @ self.weight.t()

        elif self.backend == "2:4_magnitude_scale":
            x_flat_sp, alpha_scale = self.magnitude_pruner_scale(x_flat, self.weight, prune_n=2, prune_m=4)
            out = x_flat_sp @ ((self.weight * alpha_scale).t())
            # out = x_flat_sp @ self.weight.t()

        elif self.backend == "2:4_magnitude_hadamard":
            x_flat_sp = self.magnitude_pruner_hadamard(x_flat.float(), prune_n=2, prune_m=4)
            # w_transformed = hadamard_transform(self.weight.float(), scale=1.0)
            # num_cols = self.weight.shape[1]
            # out = x_flat_sp @ w_transformed.t() / num_cols
            # out = out.half()

            num_cols = self.weight.shape[1]
            out = x_flat_sp @ self.weight.float().t() / num_cols
            out = out.half()           


        elif self.backend == "2:4_variance_correction":
            # x_flat_sp = self.magnitude_var_correction(x_flat, self.weight, prune_n=2, prune_m=4)
            x_flat_sp = self.magnitude_weight_var_correction(x_flat, self.weight, prune_n=2, prune_m=4)
            out = x_flat_sp @ self.weight.t()

        elif self.backend == "2:4_weight_metric_correction":
            # x_flat_sp = self.magnitude_var_correction(x_flat, self.weight, prune_n=2, prune_m=4)
            x_flat_sp = self.magnitude_weight_metric_correction(x_flat, self.weight, prune_n=2, prune_m=4)
            out = x_flat_sp @ self.weight.t()

        elif self.backend == "cutlass":
            x_flat_sp = SparseSemiStructuredTensorCUTLASS.prune_dense_static_sort(x_flat)
            out = x_flat_sp @ self.weight.t()

        elif self.backend == "8:16_magnitude":
            x_flat_sp = self.magnitude_pruner(x_flat, prune_n=8, prune_m=16)
            out = x_flat_sp @ self.weight.t()

        elif self.backend == "8:16_variance_correction":
            # x_flat_sp = self.magnitude_var_correction(x_flat, self.weight, prune_n=2, prune_m=4)
            x_flat_sp = self.magnitude_weight_var_correction(x_flat, self.weight, prune_n=8, prune_m=16)
            out = x_flat_sp @ self.weight.t()

        elif self.backend == "8:16_variance_correction_outliers":
            # x_flat_sp = self.magnitude_var_correction(x_flat, self.weight, prune_n=2, prune_m=4)
            x_flat_sp = self.magnitude_weight_outliers_var_correction(x_flat, self.weight, prune_n=8, prune_m=16)
            out = x_flat_sp @ self.weight.t()



        out = out.view(bs, seq_len, -1)

        return out

    @classmethod
    def from_original(cls, orig_linear, backend=None, name=None):
        linear_sp = cls(
            orig_linear.in_features, 
            orig_linear.out_features, 
            backend=backend
        )
        linear_sp.weight = orig_linear.weight
        linear_sp.name = name

        return linear_sp

