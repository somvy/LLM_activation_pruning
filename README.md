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



## Qwen2.5-7B-Instruct
*(Note: with chat_template)*

| Pruning | wikitext2 (ppl) | arc easy | boolq | piqa | winogrande | runtime (min) | replaced linear layers |
|---------|-----------------|----------|-------|------|------------|---------------|------------------------|
| none    | 7.458          | 0.686    | 0.858 | 0.747| 0.603      | 9             | 0                      |
| 2:4     | 237619.203     | 0.255    | 0.4385| 0.527| 0.501      | 22            | 196                    |
| 8:16    | 106531         | 0.263    | 0.413 | 0.5212| 0.4996    | 15.5          | 196                    |

## Gemma3-4b-it
*(Note: with chat_template)*

| Pruning | wikitext2 (ppl) | arc easy | boolq | piqa | winogrande | runtime (min) | replaced linear layers |
|---------|-----------------|----------|-------|------|------------|---------------|------------------------|
| none    | 17.29          | 0.720    | 0.840 | 0.721| 0.617      | 21            | 0                      |
| 2:4     | 35.62          | 0.654    | 0.758 | 0.697| 0.506      | 30.5          | 319                    |
| 8:16    | 25.31          | 0.700    | 0.81  | 0.712| 0.554      | 29            | 319                    |
