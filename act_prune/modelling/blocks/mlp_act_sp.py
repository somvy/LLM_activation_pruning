import torch
from torch import nn

from torch.sparse import (
    SparseSemiStructuredTensorCUTLASS, 
    SparseSemiStructuredTensorCUSPARSELT
)
from transformers.activations import ACT2FN

class MLP_act_sp(nn.Module):
    def __init__(self, config, backend=None):
        super().__init__()
        if hasattr(config, "mlp_bias"):
            bias = config.mlp_bias
        else:
            bias = False
        self.backend = backend

        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.backend is None:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        
        elif self.backend == "cusparselt":
            bs, seq_len, _ = x.shape
            x_flat = x.view(-1, self.hidden_size)
            pruned_x = SparseSemiStructuredTensorCUSPARSELT.prune_dense_static_sort(x_flat)
            down_proj = self.down_proj(self.act_fn(self.gate_proj(pruned_x)) * self.up_proj(pruned_x))
        
        elif self.backend == "cutlass":
            bs, seq_len, _ = x.shape
            x_flat = x.view(-1, self.hidden_size)
            pruned_x = SparseSemiStructuredTensorCUTLASS.prune_dense_static_sort(x_flat)
            out_gate_proj = self.gate_proj(x_flat).view(bs, seq_len, -1)
            out_up_proj = self.up_proj(x_flat).view(bs, seq_len, -1)
            down_proj = self.down_proj(self.act_fn(out_gate_proj) * out_up_proj)  

        return down_proj
    
    @classmethod
    def from_original(cls, orig_MLP, backend=None):
        mlp_sp = cls(orig_MLP.config, backend)
        mlp_sp.gate_proj = orig_MLP.gate_proj
        mlp_sp.up_proj = orig_MLP.up_proj
        mlp_sp.down_proj = orig_MLP.down_proj

        return mlp_sp