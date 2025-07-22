import logging
from pathlib import Path
import json
import torch
from base_runner import BaseRunner
from modelling.blocks.llama_attn import LlamaAttention_act_sp
from modelling.blocks.mlp_act_sp import MLP_act_sp
from modelling.layers.linear_act_sp import Linear_act_sp


def weight_prune(layer, sparsity_type, sparsity_ratio: float, prune_n, prune_m, name):
    w = layer.weight.data
    b = layer.bias.data if layer.bias is not None else None

    if sparsity_type == "semi-structured_weight_magnitude":
        orig_shape = w.shape
        w_1d = w.view(-1, prune_m)

        _, idx = torch.topk(w_1d.abs(), prune_n, dim=1, sorted=False)
        mask_1d = torch.zeros_like(w_1d)
        mask_1d.scatter_(dim=1, index=idx, value=True)
        mask = mask_1d.view(orig_shape)
        pruned_w = w * mask

    elif sparsity_type == "unstructured_weight_magnitude":
        w_1d = w.abs().flatten()
        k = int(len(w_1d) * sparsity_ratio)
        thresh = torch.kthvalue(w_1d, k)[0]
        mask = w.abs() >= thresh
        pruned_w = w * mask
    else:
        raise ValueError(f"Unsupported sparsity type: {sparsity_type}")


    new_layer = torch.nn.Linear(layer.in_features, layer.out_features, bias=layer.bias is not None)
    new_layer.weight.data = pruned_w

    if b is not None:
        new_layer.bias.data = b.clone()

    new_layer.name = name
    return new_layer
class ActPruneRunner(BaseRunner):

    def __init__(self, config):
        super().__init__(config)

    def replace_linear_layers(self):
        """Insert into model modified linear layers with original weights"""
        logging.info("Replace Linear layers...")
        # architectures = self.model.config["architectures"]
        architectures = self.model.config.architectures

        if self.config["model"]["path"] == "google/gemma-3-4b-it":
            root_module = self.model
        else:
            root_module = self.model.model
        module_name_dict = {name: module for name, module in root_module.named_modules()}

        sparsity_type = self.config["pruning"]["sparsity_type"]
        sparsity_ratio = self.config["pruning"].get("sparsity_ratio", None)
        prune_n = self.config["pruning"].get("prune_n", None)
        prune_m = self.config["pruning"].get("prune_m", None)
        target_layers = self.config["pruning"]["target_modules"]

        replaced_cnt = 0
        for name, module in module_name_dict.items():
            if isinstance(module, torch.nn.Linear):
                ind = name.rfind(".")
                if ind == -1:
                    father = module_name_dict[""]
                else:
                    father = module_name_dict[name[:ind]]

                if name[(ind+1):] in target_layers:
                    kvargs = dict(
                        sparsity_type=sparsity_type,
                        sparsity_ratio=sparsity_ratio,
                        prune_n=prune_n,
                        prune_m=prune_m,
                        name=name[(ind + 1) :],
                    )
                    if sparsity_type in ("semi-structured_act_magnitude","unstructured_act_magnitude"):
                        sparse_linear = Linear_act_sp.from_original(module, **kvargs)
                    elif sparsity_type in ("semi-structured_weight_magnitude", "unstructured_weight_magnitude"):
                        sparse_linear = weight_prune(module, **kvargs)
                    else:
                        raise ValueError(f"Unsupported sparsity type: {sparsity_type}")

                    setattr(father, name[ind + 1 :], sparse_linear)
                    replaced_cnt += 1
                    logging.info(name)

        logging.info("Modified model...")
        logging.info(self.model)
        logging.info(f"Replaced {replaced_cnt} linear layers with sparse ones.")

    def replace_mlp_blocks(self):
        """Insert into model modified mlp blocks with original weights"""

        logging.info("Replace MLP blocks...")
        # architectures = self.model.config["architectures"]
        architectures = self.model.config.architectures
        orig_mlp_block = self.model.model.layers[0].mlp
        root_module = self.model.model

        sparsity_type = self.config["pruning"]["sparsity_type"]
        target_layers = self.config["pruning"]["target_modules"]
        module_name_dict = {name: module for name, module in root_module.named_modules()}
        replaced_cnt = 0
        for name, module in module_name_dict.items():
            if isinstance(module, type(orig_mlp_block)):
                ind = name.rfind(".")
                if ind == -1:
                    father = module_name_dict[""]
                else:
                    father = module_name_dict[name[:ind]]

                sparse_mlp = MLP_act_sp.from_original(module, sparsity_type=sparsity_type)
                setattr(father, name[ind + 1 :], sparse_mlp)
                replaced_cnt += 1
                logging.info(name)

        logging.info("Modified model...")
        logging.info(self.model)
        logging.info(f"Replaced {replaced_cnt} MLP blocks with sparse ones.")

    def replace_attn_blocks(self):
        """Insert into model modified self attn blocks with original weights"""

        logging.info("Replace SelfAttn blocks...")
        # architectures = self.model.config["architectures"]
        architectures = self.model.config.architectures
        orig_self_attn_block = self.model.model.layers[0].self_attn
        root_module = self.model.model

        sparsity_type = self.config["pruning"]["sparsity_type"]
        target_layers = self.config["pruning"]["target_modules"]
        module_name_dict = {name: module for name, module in root_module.named_modules()}
        replaced_cnt = 0
        for name, module in module_name_dict.items():
            if isinstance(module, type(orig_self_attn_block)):
                ind = name.rfind(".")
                if ind == -1:
                    father = module_name_dict[""]
                else:
                    father = module_name_dict[name[:ind]]

                sp_SelfAttn = LlamaAttention_act_sp.from_original(
                    module, sparsity_type=sparsity_type
                )
                setattr(father, name[ind + 1 :], sp_SelfAttn)
                replaced_cnt += 1
                logging.info(name)

        logging.info("Modified model...")
        logging.info(self.model)
        logging.info(f"Replaced {replaced_cnt} SelfAttn blocks with sparse ones.")

    def run(self):
        """Execute the pruning pipeline."""
        self.load_model_tokenizer()

        log_dir = Path(self.config["paths"]["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        res_file = log_dir / f"{self.run_id}.json"

        res = {}
        res["config"] = self.config
        sparsity_type = self.config["pruning"]["sparsity_type"]
        if sparsity_type == "None":
            logging.info("No sparsity applied, using original model.")
        else:
            self.replace_linear_layers()

        benchmarks = self.config["benchmarks"]

        if benchmarks["ppl_wikitext2"]["run_ppl"]:
            _, testloader = self.load_data("wikitext2")
            ppl, time = self.measure_ppl(testloader, bs=benchmarks["ppl_wikitext2"]["batch_size"])

            logging.info(f'wikitext2: {ppl}, computation time: {time}')
            res["wikitext ppl"] = ppl
            res["wikitext time"] = time


        if benchmarks["harness"]["run_lm_eval"]:
            results = self.run_lm_eval()
            logging.info(results)
            res["lm_eval"] = results

        with res_file.open("w") as f:
            json.dump(res, f, indent=2)
            logging.info("Results saved to %s", res_file)
