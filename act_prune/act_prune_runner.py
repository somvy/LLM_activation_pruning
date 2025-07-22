import logging

import torch
from base_runner import BaseRunner
from modelling.blocks.llama_attn import LlamaAttention_act_sp
from modelling.blocks.mlp_act_sp import MLP_act_sp
from modelling.layers.linear_act_sp import Linear_act_sp


class ActPruneRunner(BaseRunner):

    def __init__(self, config):
        super().__init__(config)
        # self.log_dir = (Path(config["paths"]["log_dir"]) / self.model_save_name / self.dataset_name)

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

                if name[(ind + 1) :] in target_layers:
                    sparse_linear = Linear_act_sp.from_original(
                        module,
                        sparsity_type=sparsity_type,
                        sparsity_ratio=sparsity_ratio,
                        prune_n=prune_n,
                        prune_m=prune_m,
                        name=name[(ind + 1) :],
                    )
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
        # self.model.config._attn_implementation == 'eager'
        # self.log_dir = (Path(config["paths"]["log_dir"]) / self.model_save_name / self.dataset_name)
        sparsity_type = self.config["pruning"]["sparsity_type"]
        if (
            sparsity_type == "semi-structured_act_magnitude"
            or sparsity_type == "unstructured_act_magnitude"
        ):
            if self.config["pruning"]["module"] == "layers":
                self.replace_linear_layers()
            elif self.config["pruning"]["module"] == "mlp_blocks":
                self.replace_mlp_blocks()
            elif self.config["pruning"]["module"] == "attn_blocks":
                self.replace_attn_blocks()
        elif sparsity_type == "None":
            logging.info("No sparsity applied, using original model.")
        else:
            raise ValueError(f"Unsupported sparsity type: {sparsity_type}")

        benchmarks = self.config["benchmarks"]
        if benchmarks["ppl_wikitext2"]["run_ppl"]:
            _, testloader = self.load_data("wikitext2")
            # Evaluate ppl in no grad context to avoid updating the model
            ppl, time = self.measure_ppl(testloader, bs=benchmarks["ppl_wikitext2"]["batch_size"])
            logging.info(f"wikitext2: {ppl}, computation time: {time}")

        if benchmarks["harness"]["run_lm_eval"]:
            results = self.run_lm_eval()
            logging.info(results)
