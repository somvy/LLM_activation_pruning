import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed

def get_model(model_path, seq_len):

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code = True,
        torch_dtype='auto',
        device_map = 'cuda:0'
    )
    model.seq_len = seq_len

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer