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
        sparsity_ratio=None,  # if sparsity_type is "unstructured_act_magnitude"
        prune_n=None,  # if sparsity_type is "semi-structured_act_magnitude"
        prune_m=None,  # if sparsity_type is "semi-structured_act_magnitude"
        name=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity_type = sparsity_type
        self.sparsity_ratio = sparsity_ratio
        self.prune_n = prune_n
        self.prune_m = prune_m
        self.register_buffer("weight", None)
        self.name = name

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

    def forward(self, x):
        bs, seq_len, _ = x.shape
        x_flat = x.view(-1, self.in_features)

        if self.sparsity_type is None:
            out = x @ self.weight.t()
        elif self.sparsity_type == "semi-structured_act_magnitude":
            x_flat_sp = self.semi_structural_magnitude_pruner(
                x_flat, prune_n=self.prune_n, prune_m=self.prune_m
            )
            out = x_flat_sp @ self.weight.t()
        elif self.sparsity_type == "unstructured_act_magnitude":
            x_flat_sp = self.unstructured_magnitude_pruner(
                x_flat, sparsity_ratio=self.sparsity_ratio
            )
            out = x_flat_sp @ self.weight.t()

        out = out.view(bs, seq_len, -1)

        return out

    @classmethod
    def from_original(
        cls,
        orig_linear,
        sparsity_type=None,
        sparsity_ratio=None,
        prune_n=None,
        prune_m=None,
        name=None,
    ):
        linear_sp = cls(
            orig_linear.in_features,
            orig_linear.out_features,
            sparsity_type=sparsity_type,
            sparsity_ratio=sparsity_ratio,
            prune_n=prune_n,
            prune_m=prune_m,
            name=name,
        )
        linear_sp.weight = orig_linear.weight

        return linear_sp
