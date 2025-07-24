import torch
from torch import nn

# from fast_hadamard_transform import hadamard_transform

class Linear_act_sp(nn.Module):
    def __init__(
        self, 
        in_features, 
        out_features,
        bias=False,
        sparsity_type=None,
        sparsity_ratio=None,
        prune_n = None,
        prune_m = None,
        name=None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity_type = sparsity_type
        self.sparsity_ratio = sparsity_ratio
        self.prune_n=prune_n
        self.prune_m=prune_m
        self.register_buffer('weight', None)
        self.name = name


    def semi_structural_magnitude_pruner(self, x, prune_n=2, prune_m=4):
        x_metric = torch.abs(x)

        orig_shape = x.shape
        x_1d = x.view(-1, prune_m)

        
        _, idx = torch.topk(x_1d.abs(), prune_n, dim=1, sorted=False)
        mask_1d = torch.zeros_like(x_1d)
        mask_1d.scatter_(dim=1, index=idx, value=True)
        mask = mask_1d.view(orig_shape)
        x_sp = x * mask
        return x_sp

    def semi_structural_L_pruner(self, x, prune_n=2, prune_m=4):
        """
        If we remove X_{it} from the input activation X:
            L_{cos_{ti}} = |X_it| * sqrt(sum_p X_pt^2)  / sqrt(sum_j X_ij^2)
        h is hidden dimension, l is sequence length.
        """
    
        abs_x = torch.abs(x) # |X_it|
        denominator = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))  # sqrt(sum_j X_ij^2)
        col_norms = torch.sqrt(torch.sum(x ** 2, dim=0, keepdim=True))  # sqrt(sum_p X_pt^2)
    
        L_metric = abs_x / (denominator + 1e-8) * col_norms
    
        orig_shape = L_metric.shape
        L_metric_1d = L_metric.view(-1, prune_m)
    
        _, idx = torch.topk(L_metric_1d, prune_n, dim=1, largest=False, sorted=False)
        mask_1d = torch.ones_like(L_metric_1d, dtype=torch.bool)
        mask_1d.scatter_(dim=1, index=idx, value=False)
        mask = mask_1d.view(orig_shape).view_as(x)
        x_sp = x * mask
        return x_sp

    def forward (self, x):
        bs, seq_len, _ = x.shape
        x_flat = x.view(-1, self.in_features)

        if self.sparsity_type is None:
            out = x @ self.weight.t()
        
        elif self.sparsity_type == "semi-structured_act_magnitude":
            
            x_flat_sp = self.semi_structural_magnitude_pruner(
                x_flat, 
                prune_n=self.prune_n, prune_m=self.prune_m
            )
            out = x_flat_sp @ self.weight.t()

        elif self.sparsity_type == "semi_structural_L_pruner":
            
            x_flat_sp = self.semi_structural_L_pruner(
                x_flat,
                prune_n=self.prune_n, 
                prune_m=self.prune_m
            )
            out = x_flat_sp @ self.weight.t()

        elif self.sparsity_type == "semi-structured_shift":
            eta = torch.mean(x_flat, dim=1, keepdim=True)  # eq (9)
            x_shifted = x_flat - eta
            x_sp = self.semi_structural_magnitude_pruner(
                x_shifted, 
                prune_n=self.prune_n, 
                prune_m=self.prune_m
            )
            out = (x_sp + eta) @ self.weight.t() # eq (8)

        out = out.view(bs, seq_len, -1)

        return out

    @classmethod
    def from_original(
        cls, 
        orig_linear, 
        sparsity_type=None, 
        sparsity_ratio=None, 
        prune_n=None, prune_m=None, 
        name=None
    ):
        linear_sp = cls(
            orig_linear.in_features, 
            orig_linear.out_features, 
            sparsity_type=sparsity_type,
            sparsity_ratio=sparsity_ratio,
            prune_n=prune_n, prune_m=prune_m,
            name=name
        )
        linear_sp.weight = orig_linear.weight

        return linear_sp

