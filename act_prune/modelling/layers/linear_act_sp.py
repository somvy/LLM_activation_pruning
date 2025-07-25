import torch
from torch import nn

# from fast_hadamard_transform import hadamard_transform


class Linear_act_sp(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        sparsity_type=None,  # [None, "semi-structured_act_magnitude", "unstructured_act_magnitude"]
        transformation_type=None,
        sparsity_ratio=None,  # if sparsity_type is "unstructured_act_magnitude"
        prune_n=None,  # if sparsity_type is "semi-structured_act_magnitude"
        prune_m=None,  # if sparsity_type is "semi-structured_act_magnitude"
        name=None,
        additional_transformation=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity_type = sparsity_type
        self.transformation_type = transformation_type
        self.sparsity_ratio = sparsity_ratio
        self.prune_n = prune_n
        self.prune_m = prune_m
        self.register_buffer("weight", None)
        self.name = name
        self.additional_transformation = additional_transformation

    def unstructured_magnitude_pruner(self, x, sparsity_ratio):
        orig_shape = x.shape
        num_elements_to_keep = int(orig_shape[1] * (1.0 - sparsity_ratio))

        _, idx = torch.topk(x.abs(), num_elements_to_keep, dim=1, sorted=False)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask.scatter_(dim=1, index=idx, value=True)
        x_sp = x * mask
        return x_sp

    def semi_structural_magnitude_pruner(self, x, prune_n=2, prune_m=4):
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

    def prune_with_additional_transformation(self, x, pruner):
        if self.additional_transformation == "scaling":
            return self.scaling_transformation(x, pruner)
        return pruner(x) @ self.weight.t()

    def forward (self, x):
        bs, seq_len, _ = x.shape
        x_flat = x.view(-1, self.in_features)
        out = None

        # Without pruning
        if self.sparsity_type is None:
            out = x @ self.weight.t()

        # Semi-structured with transformation logic
        elif self.sparsity_type == "semi-structured_act_magnitude":
            pruner = lambda z: self.semi_structural_magnitude_pruner(z, self.prune_n, self.prune_m)

            if self.transformation_type == "variance":
                x_sp = pruner(x_flat)
                out = self.variance_transformation(x_flat, x_sp) @ self.weight.t()
            elif self.transformation_type == "shift":
                out = self.shift_transformation(x_flat, pruner, self.bias_term(x_flat)) @ self.weight.t()
            elif self.transformation_type == "learnable":
                out = self.learnable_transformation(x_flat, pruner) @ self.weight.t()
            elif self.transformation_type == "scaling" or self.additional_transformation == "scaling":
                out = self.scaling_transformation(x_flat, pruner)
            else:
                out = pruner(x_flat) @ self.weight.t()
        
        # Unstructured pruning
        elif self.sparsity_type == "unstructured_act_magnitude":
            pruner = lambda z: self.unstructured_magnitude_pruner(z, self.sparsity_ratio)
            out = self.prune_with_additional_transformation(x_flat, pruner)

        # L-based pruning from shirin-shift-transform
        elif self.sparsity_type == "semi_structural_L_pruner":
            x_sp = self.semi_structural_L_pruner(x_flat, self.prune_n, self.prune_m)
            out = x_sp @ self.weight.t()

        # Shift-only variant from shirin
        elif self.sparsity_type == "semi-structured_shift":
            eta = self.bias_term(x_flat)
            x_shifted = x_flat - eta
            x_sp = self.semi_structural_magnitude_pruner(x_shifted, self.prune_n, self.prune_m)
            out = (x_sp + eta) @ self.weight.t()

        else:
            raise ValueError(f"Unknown sparsity_type: {self.sparsity_type}")

        return out.view(bs, seq_len, -1)

    @classmethod
    def from_original(
        cls,
        orig_linear,
        sparsity_type=None,
        sparsity_ratio=None,
        transformation_type=None,
        prune_n=None,
        prune_m=None,
        name=None,
        additional_transformation=None,
    ):
        linear_sp = cls(
            orig_linear.in_features,
            orig_linear.out_features,
            sparsity_type=sparsity_type,
            transformation_type=transformation_type,
            sparsity_ratio=sparsity_ratio,
            prune_n=prune_n,
            prune_m=prune_m,
            name=name,
            additional_transformation=additional_transformation,
        )
        linear_sp.weight = orig_linear.weight

        return linear_sp
