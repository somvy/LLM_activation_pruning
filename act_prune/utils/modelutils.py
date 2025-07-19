import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
import os


def get_model(model_path, seqlen):

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code = True,
        torch_dtype='auto',
        device_map = 'cuda:0',
        # attn_implementation = 'eager'
    )
    model.seqlen = seqlen

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer