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
        transformation_type=None,
        sparsity_ratio=None,
        prune_n = None,
        prune_m = None,
        name=None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity_type = sparsity_type
        self.transformation_type = transformation_type
        self.sparsity_ratio = sparsity_ratio
        self.prune_n = prune_n
        self.prune_m = prune_m
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

    def variance_factor(self, x, x_sp):
        var_ratio = torch.var(x) / torch.clamp(torch.var(x_sp), min=1e-9)
        v = torch.sqrt(var_ratio)
        return v
    
    def variance_transformation(self, x, x_sp):
        v = self.variance_factor(x, x_sp)
        corr_x_sp = v * x_sp
        return corr_x_sp

    def bias_term(self, x):
        eta = torch.mean(x, dim=1, keepdim=True)
        return eta

    def shift_transformation(self, x, pruner, eta):
        x_shifted = x - eta
        x_sp = pruner(x_shifted)
        x_sp_shifted = x_sp + eta
        return x_sp_shifted

    def scaling_transformation(self, x, pruner):
        max_act = torch.max(torch.abs(x), dim=0).values
        max_weight = torch.max(torch.abs(self.weight), dim=0).values
        s = torch.sqrt(max_act / max_weight.clamp(min=1e-8))
        x_flat_sp = pruner(x / s)
        scaled_weight = self.weight * s.unsqueeze(0)
        return x_flat_sp @ scaled_weight.t()
        
    def learnable_transformation(self, x, pruner):
        if not hasattr(self, 'eta'):
            self.eta = nn.Parameter(self.bias_term(x))
            
        bs = x.shape[0]
        x_sp_shifted = self.shift_transformation(x, pruner, self.eta[:bs])

        if not hasattr(self, 'v'):
            x_sp = pruner(x)
            self.v = nn.Parameter(self.variance_factor(x, x_sp))
        
        corr_x_sp_shifted = self.v * x_sp_shifted
        return corr_x_sp_shifted
        
    def forward (self, x):
        bs, seq_len, _ = x.shape
        x_flat = x.view(-1, self.in_features)
        out = None

        if self.sparsity_type is None:
            out = x @ self.weight.t()
        
        elif self.sparsity_type == "semi-structured_act_magnitude":
            
            x_flat_sp = self.semi_structural_magnitude_pruner(
                x_flat, 
                prune_n=self.prune_n, prune_m=self.prune_m
            )

            if self.transformation_type is None:
                out = x_flat_sp @ self.weight.t()

            elif self.transformation_type == "variance":
                corr_x_flat_sp = self.variance_transformation(
                    x_flat, x_flat_sp
                )
                out = corr_x_flat_sp @ self.weight.t()

            elif self.transformation_type == "shift":
                pruner = lambda x: self.semi_structural_magnitude_pruner(
                    x, prune_n=self.prune_n, prune_m=self.prune_m
                )
                eta = self.bias_term(x_flat)
                x_sp_shifted = self.shift_transformation(
                    x_flat, pruner, eta
                )
                out = x_sp_shifted @ self.weight.t()

            elif self.transformation_type == "scaling":
                pruner = lambda x: self.semi_structural_magnitude_pruner(
                    x, prune_n=self.prune_n, prune_m=self.prune_m
                )
                out = self.scaling_transformation(
                    x_flat, pruner
                )

            elif self.transformation_type == "learnable":
                pruner = lambda x: self.semi_structural_magnitude_pruner(
                    x, prune_n=self.prune_n, prune_m=self.prune_m
                )
                corr_x_sp_shifted = self.learnable_transformation(
                    x_flat, pruner
                )
                out = corr_x_sp_shifted @ self.weight.t()


        out = out.view(bs, seq_len, -1)

        return out

    @classmethod
    def from_original(
        cls, 
        orig_linear, 
        sparsity_type=None,
        transformation_type=None,
        sparsity_ratio=None, 
        prune_n=None, prune_m=None, 
        name=None
    ):
        linear_sp = cls(
            orig_linear.in_features, 
            orig_linear.out_features, 
            sparsity_type=sparsity_type,
            transformation_type=transformation_type,
            sparsity_ratio=sparsity_ratio,
            prune_n=prune_n, prune_m=prune_m,
            name=name
        )
        linear_sp.weight = orig_linear.weight

        return linear_sp

