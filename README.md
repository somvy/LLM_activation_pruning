# LLM_activation_pruning
This repository implements activation-level pruning for speeding up inference and fine-tuning in large language models (LLMs) such as LLaMA-2. The pruning is applied to the input activations of linear layers using structured and unstructured methods, with support for learnable transformations.

# 🚀 Quickstart

## 🧠 Project Structure

```
LLM_activation_pruning/
├── act_prune/
│   ├── main.py                  # Entry point
│   ├── act_prune_runner.py      # Runner logic
│   ├── base_runner.py           # Abstract runner with model/data loading
│   ├── configs/config.yaml      # Config file
│   ├── run_exps.sh              # Batch experiment runner
│   ├── modelling/               # Custom blocks and layers
│   ├── utils/                   # Data and model helpers
├── notebooks/                   
├── README.md                    
```

## 📦 Environment
We recommend using PyTorch = 2.6.0 with CUDA 12.4. 
Additionally, you need to install the following packages:
```
pip install transformers==4.51.3
pip install lm_eval==0.4.9
pip install yq
```
Install any other required packages as needed.

## ⚙️ Parameters
All experiment settings are defined in `act_prune/configs/config.yaml`. Here's a breakdown:

```yaml
env:
  SEED: 42
  CUDA_DEVICE_ORDER: PCI_BUS_ID
  OMP_NUM_THREADS: 4

paths:
  data_dir: data/
  log_dir: artifacts/logs/
  checkpoint_dir: artifacts/checkpoints/
  results_dir: artifacts/results/

model:
  path: meta-llama/Llama-2-7b
  seqlen: 2048

benchmarks:
  ppl_wikitext2:
    run_ppl: true
    batch_size: 8
  harness:
    run_lm_eval: true
    tasks: [boolq, winogrande, piqa, arc_easy]
    num_fewshot: 0
    batch_size: 1024
    apply_chat_template: false

pruning:
  sparsity_type: semi-structured_act_magnitude
  transformation_type: learnable
  sparsity_ratio: 0.2
  additional_transformation: scaling
  prune_n: 8
  prune_m: 16
  module: layers
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
```

## Experiments
To conduct an experiment, simply run the code from the directory `./act_prune`:

```python
python main.py
```

To run a bunch of experiments, use the run_exps.sh script: (install yq if not)

```bash
bash run_exps.sh
```


## 📊 Experiment Results


### Llama2-7b

*(Note: no chat_template, some how it is missing in NousResearch/Llama-2-7b-chat-hf )*

| Pruning | wikitext2 (ppl) | arc easy | boolq | piqa | winogrande | runtime (min) | replaced linear layers |
|---------|-----------------|----------|-------|------|------------|---------------|------------------------|
| none    |  6.943          |0.738     |0.797  |0.764 |0.663       | 9             | 0                      |
| 2:4     | 10.23           |0.664     |0.706  |0.714 |0.604       | 20            | 224                    |
| 8:16    | 8.1238          | 0.692    |0.752  |0.731 |0.630       | 16.5          | 224                    |
| 2:4 wt  | 42.4            | 0.571    |0.648  |0.686 | 0.558      | 9             | 224                    |
| 8:16 wt | 20.47           | 0.641    |0.757  |0.720 | 0.606      | 9             | 224                    |
| 8:16 wt | 13.39           | 0.675    |0.640  |0.729 | 0.649      |               | 224                    |
| 0.2 | - | 0.735 | 0.800 | 0.769 | 0.664 | | |
| 0.5 | - | 0.704 | 0.776 | 0.754 | 0.661 | | |
| 0.7 | 20.11 | 0.559 | 0.639 | 0.652 | 0.525 | | |
| 0.9 | 6867.29 | 0.256 | 0.379 | 0.523 | 0.490 | | |
| 0.25 wt | 5.884           | 0.765    |0.755  |0.782 | 0.706      |               | 224                    |
| 0.5  wt | 15.68           | 0.649    |0.619  |0.721 | 0.636      |               | 224                    |
| 0.75 wt | 63609.81        | 0.258    |0.405  |0.534 | 0.499      |               | 224                    |


## Qwen2.5-7B-Instruct
*(Note: with chat_template)*

| Pruning | wikitext2 (ppl) | arc easy | boolq | piqa | winogrande | runtime (min) | replaced linear layers |
|---------|-----------------|----------|-------|------|------------|---------------|------------------------|
| none    | 7.458           | 0.686    | 0.858 | 0.747| 0.603      | 9             | 0                      |
| 2:4     | 237619.203      | 0.255    | 0.438 | 0.527| 0.501      | 22            | 196                    |
| 8:16    | 106531          | 0.263    | 0.413 | 0.521| 0.499      | 15.5          | 196                    |
| 2:4 wt  | 5994.70         | 0.375    | 0.661 | 0.558| 0.495      | | 196 |
| 8:16 wt | 7890.85         | 0.489    | 0.749 | 0.557| 0.510      | | 196|
| 0.2 | 32158.26 | 0.261 | 0.404 | 0.524 | 0.501 | | |
| 0.5 | 32379.28 | 0.250 | 0.429 | 0.522 | 0.498 | | |
| 0.7 | 3238375.75 | 0.257 | 0.429 | 0.534 | 0.494 | | |
| 0.9 | 38221328.0 | 0.261 | 0.378 | 0.534 | 0.507 | | |
| 0.25 wt | 10.09           | 0.664    |0.858  |0.734 | 0.609      |               | 196                    |
| 0.5  wt | 440.05          | 0.539    |0.738  |0.633 | 0.541      |               | 196                    |
| 0.75 wt | 49425.35        | 0.262    |0.404  |0.525 | 0.485      |               | 196                    |
|0.7 ACT(qkv) |   19.28         | 0.584    |0.78   | 0.68 | 0.55       |               | 112                    |
|0.2 ACT(qkvo)|    7.48         | 0.69     |0.86   | 0.74 | 0.61       |               | 84                     |
|0.5 ACT(qkvo)|    8.29         | 0.67     |0.87   | 0.74 | 0.58       |               | 84                     |
|0.7 ACT(qkvo)|   18.69         | 0.605    |0.81   | 0.70 | 0.58       |               | 84                     |
|0.9 ACT(qkvo)|   4270.93       | 0.25     |0.38   | 0.54 | 0.52       |               | 84                     |
|0.2 WT (qkvo)|    8.03         | 0.67     |0.86   | 0.74 | 0.60       |               | 84                     |
|0.5 WT (qkvo)|    43.56        | 0.56     |0.80   | 0.68 | 0.57       |               | 84                     |
|0.7 WT (qkvo)|    1867.03      | 0.28     |0.38   | 0.54 | 0.48       |               | 84                     |
|0.9 WT (qkvo)|   50451.45      | 0.25     |0.58   | 0.54 | 0.51       |               | 84                     |
|8:16 ACT (qkvo)|    8.61      | 0.66     |0.87   | 0.73 | 0.55       |               |                     |
|2:4  ACT (qkvo)|   10.06      | 0.65     |0.86   | 0.72 | 0.54       |               |                     |
|8:16 WT (qkvo)|    40.79      | 0.59     |0.82   | 0.67 | 0.55        |               |                     |
|2:4  WT (qkvo)|    35.37      | 0.53     |0.78   | 0.68 | 0.54       |               |                     |
|8:16 PTS (qkvo)|   8.61       | 0.80     |0.87  | 0.77 | 0.68       |               |                     |
|2:4  PTS (qkvo)|  10.07       | 0.79     | 0.86  | 0.76 | 0.66       |               |                     |
|8:16 VAR (qkvo)|   8.44       | 0.77     |0.85   | 0.73 | 0.68       |               |                     |
|2:4  VAR (qkvo)|   9.51       | 0.77     |0.85  | 0.73 | 0.66       |               |                     |

## Gemma3-4b-it
*(Note: with chat_template)*

| Pruning | wikitext2 (ppl) | arc easy | boolq | piqa | winogrande | runtime (min) | replaced linear layers |
|---------|-----------------|----------|-------|------|------------|---------------|------------------------|
| none    | 17.29           | 0.720    | 0.840 | 0.721| 0.617      | 21            | 0                      |
| 2:4     | 35.62           | 0.654    | 0.758 | 0.697| 0.506      | 30.5          | 319                    |
| 8:16    | 25.31           | 0.700    | 0.81  | 0.712| 0.554      | 29            | 319                    |
| 2:4 wt  | 421.95          | 0.347    | 0.441 | 0.577| 0.493      | | 319|
| 8:16 wt | 198.53          | 0.391    | 0.601 | 0.618| 0.515      | | 319|
| 0.2     | 17.60 | 0.714 | 0.843 | 0.717 | 0.603 | | |
| 0.5     | 22.39 | 0.707 | 0.825 | 0.717 | 0.571 | | |
| 0.7     | 88.68 | 0.550 | 0.634 | 0.660 | 0.535 | | |
| 0.9     | 214007.73 | 0.263 | 0.378 | 0.541 | 0.498 | | |

wt stands for  weight pruning
