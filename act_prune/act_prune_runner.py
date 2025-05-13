import logging
import os
import json
import pickle
import torch
from pathlib import Path

from base_runner import BaseRunner
from modelling.blocks.mlp_act_sp import MLP_act_sp

class ActPruneRunner(BaseRunner):

    def __init__(self, config):
        super().__init__(config)
        # self.log_dir = (Path(config["paths"]["log_dir"]) / self.model_save_name / self.dataset_name)

    def replace_mlp_blocks(self, backend=None):
        """Insert into model modified mlp blocks with original weights """

        logging.info("Replace MLP blocks...")
        # architectures = self.model.config["architectures"]
        architectures = self.model.config.architectures
        orig_mlp_block = self.model.model.layers[0].mlp
        root_module = self.model.model
        module_name_dict = {name: module for name, module in root_module.named_modules()}
        for name, module in module_name_dict.items():
            # if isinstance(module, Qwen3MLP):
            # if isinstance(module, LlamaMLP):
            # if isinstance(module, torch.nn.Linear) and (name.find("down_proj") != -1):
            if isinstance(module, type(orig_mlp_block)):
                ind = name.rfind(".")
                if ind == -1:
                    father = module_name_dict[""]
                else:
                    father = module_name_dict[name[:ind]]
                
                sparse_mlp = MLP_act_sp.from_original(module, backend=backend)
                setattr(father, name[ind + 1 :], sparse_mlp)
                logging.info(name)

        logging.info("Modified model...")
        logging.info(self.model)

    def run(self):
        """Execute the pruning pipeline."""
        self.load_model_tokenizer()

        # self.log_dir = (Path(config["paths"]["log_dir"]) / self.model_save_name / self.dataset_name)
        self.replace_mlp_blocks(backend=self.config["pruning"]["backend"])
        _, testloader = self.load_data()

        # Evaluate ppl in no grad context to avoid updating the model
        ppl, time = self.measure_ppl(testloader)
        
        logging.info(f'{self.config["dataset"]["name"]}: {ppl}, computation time: {time}')
