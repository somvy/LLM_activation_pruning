# LLM_activation_pruning
Code for pruning of input activations to speed up of LLMs

## Environment
To set up the environment, use the Docker container for PyTorch 2.6.0 with CUDA 12.4.
Additionally, you need to install the following packages:
```
pip install transformers==4.51.3
pip install lm_eval==0.4.9
```
Install any other required packages as needed.

## Parameters
All parameters for the experiments are contained in the configuration file located at
`./act_prune/config/config.yaml`

## Experiments
To conduct an experiment, simply run the code from the directory `./act_prune`:

```
python main.py
```

To run a bunch of experiments, use the run_exps.sh script: (install yq if not)
```
bash run_exps.sh
```


## Experiment Results


### Llama2-7b

*(Note: no chat_template, some how it is missing in NousResearch/Llama-2-7b-chat-hf )*

| Pruning | wikitext2 (ppl) | arc easy | boolq | piqa | winogrande | runtime (min) | replaced linear layers |
|---------|-----------------|----------|-------|------|------------|---------------|------------------------|
| none    |  6.943          |0.738     |0.797  |0.764 |0.663       | 9             | 0                      |
| 2:4     | 10.23           |0.664     |0.706  |0.714 |0.604       | 20            | 224                    |
| 8:16    | 8.1238          | 0.692    |0.752  |0.731 |0.630       | 16.5          | 224                    |
| 2:4 wt  | 42.4            | 0.571    |0.648  |0.686 | 0.558      | 9             | 224                    |
| 8:16 wt | 20.47           | 0.641    |0.757  |0.720 | 0.606      | 9             | 224                    |


## Qwen2.5-7B-Instruct
*(Note: with chat_template)*

| Pruning | wikitext2 (ppl) | arc easy | boolq | piqa | winogrande | runtime (min) | replaced linear layers |
|---------|-----------------|----------|-------|------|------------|---------------|------------------------|
| none    | 7.458           | 0.686    | 0.858 | 0.747| 0.603      | 9             | 0                      |
| 2:4     | 237619.203      | 0.255    | 0.438 | 0.527| 0.501      | 22            | 196                    |
| 8:16    | 106531          | 0.263    | 0.413 | 0.521| 0.499      | 15.5          | 196                    |
| 2:4 wt  | 5994.70         | 0.375    | 0.661 | 0.558| 0.495      | | 196 |
| 8:16 wt | 7890.85         | 0.489    | 0.749 | 0.557| 0.510      | | 196|

## Gemma3-4b-it
*(Note: with chat_template)*

| Pruning | wikitext2 (ppl) | arc easy | boolq | piqa | winogrande | runtime (min) | replaced linear layers |
|---------|-----------------|----------|-------|------|------------|---------------|------------------------|
| none    | 17.29           | 0.720    | 0.840 | 0.721| 0.617      | 21            | 0                      |
| 2:4     | 35.62           | 0.654    | 0.758 | 0.697| 0.506      | 30.5          | 319                    |
| 8:16    | 25.31           | 0.700    | 0.81  | 0.712| 0.554      | 29            | 319                    |
| 2:4 wt  | 421.95          | 0.347    | 0.441 | 0.577| 0.493      | | 319|
| 8:16 wt | 198.53          | 0.391    | 0.601 | 0.618| 0.515      | | 319|

wt stands for  weight pruning
