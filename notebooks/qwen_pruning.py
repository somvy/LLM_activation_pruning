import torch
from torch import nn
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensorCUTLASS, SparseSemiStructuredTensorCUSPARSELT
from torch.sparse._semi_structured_conversions import _sparse_semi_structured_tile
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP
from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.activations import ACT2FN
from datasets import load_dataset


# from torchao.dtypes.floatx import to_scaled_tc_floatx
# from torchao.ops import quant_llm_linear


def time_pytorch_function(func, input):
    # Функция для имерения скорости расчета `func` для входа `input`

    # CUDA IS ASYNC so can't use python time module
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(5):
        func(*input)

    start.record()
    func(*input)
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end)

def print_memory():
    # Функция измерения затраченной GPU памяти
    device='cuda'
    mem_allocated = torch.cuda.memory_allocated(device=device) / 1024**3
    mem_reserved = torch.cuda.memory_allocated(device=device) / 1024**3
    print(f"allocated: {mem_allocated:,.2f} gb")
    print(f" reserved: {mem_reserved:,.2f} gb")


import random

# Load and process wikitext2 dataset
def get_wikitext2(nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    # Load test datasets
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    trainloader = None
    return trainloader, testenc


# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()


# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(model, tokenizer, device=torch.device("cuda:0")):
    # Set dataset
    dataset = "wikitext2"
    model.seqlen = 2048

    # Print status
    print(f"evaluating on {dataset}")

    # Get the test loader
    _, testloader = get_wikitext2(seqlen=model.seqlen, tokenizer=tokenizer)

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl_test = eval_ppl_wikitext(model, testloader, 1, device)
    return ppl_test


def replace_mlp_blocks(root_module):

    module_name_dict = {name: module for name, module in root_module.named_modules()}
    for name, module in module_name_dict.items():
        # if isinstance(module, Qwen3MLP):
        if isinstance(module, LlamaMLP)
        # if isinstance(module, MLP_act_sp):
        # if isinstance(module, torch.nn.Linear) and (name.find("down_proj") != -1):
            ind = name.rfind(".")
            if ind == -1:
                father = module_name_dict[""]
            else:
                father = module_name_dict[name[:ind]]
            
            sparse_mlp = MLP_act_sp.from_original(module, backend="cutlass")
            setattr(father, name[ind + 1 :], sparse_mlp)

            print(f"replace mlp {name}")

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
            # out_gate_proj_sp = self.gate_proj(pruned_x)
            # down_proj = self.down_proj(self.act_fn(self.gate_proj(pruned_x)) * self.up_proj(pruned_x))

            out_gate_proj = self.gate_proj(x_flat).view(bs, seq_len, -1)
            out_up_proj = self.up_proj(x_flat).view(bs, seq_len, -1)
            down_proj = self.down_proj(self.act_fn(out_gate_proj) * out_up_proj)  

            # in_down_proj = self.act_fn(out_gate_proj) * out_up_proj
            # in_down_proj = in_down_proj.view(-1, self.intermediate_size)
            # pruned_x = SparseSemiStructuredTensorCUTLASS.prune_dense_static_sort(in_down_proj)
            # down_proj = self.down_proj(pruned_x)             
            # down_proj = down_proj.view(bs, seq_len, self.hidden_size)

        return down_proj
    
    @classmethod
    def from_original(cls, orig_MLP, backend=None):
        mlp_sp = cls(orig_MLP.config, backend)
        mlp_sp.gate_proj = orig_MLP.gate_proj
        mlp_sp.up_proj = orig_MLP.up_proj
        mlp_sp.down_proj = orig_MLP.down_proj

        return mlp_sp

def main():


    model_path = "/home/LLaMA/huggingface/Qwen3-14B"

    # model_path = "/home/LLaMA/huggingface/Qwen3-14B"
    # config = AutoConfig.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code = True,
        device_map = 'cuda:0'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    replace_mlp_blocks(model.model)

    ppl = eval_ppl(model, tokenizer)
    print(ppl)

if __name__ == "__main__":
    main()

