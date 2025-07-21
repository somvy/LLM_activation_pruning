#! /bin/bash

echo `which python`

models=(
    "NousResearch/Llama-2-7b-chat-hf"
    "Qwen/Qwen2.5-7B-Instruct"
    "google/gemma-3-4b-it"
)

for model in "${models[@]}"; do
    yq ".model.path = \"${model}\"" -i configs/config.yaml

    # run baseline
    echo "Running baseline for model: ${model}"
    yq -i ".pruning.sparsity_type = \"None\"" configs/config.yaml

    python main.py

    echo "Running structured pruning 2:4 for model: ${model}"
    yq '.pruning.sparsity_type = "semi-structured_act_magnitude"' -i configs/config.yaml
    yq ".pruning.prune_n = 2" -i configs/config.yaml
    yq ".pruning.prune_m = 4" -i configs/config.yaml
    python main.py

    echo "Running structured pruning 8:16 for model: ${model}"
    yq ".pruning.prune_n = 8" -i configs/config.yaml
    yq ".pruning.prune_m = 16" -i configs/config.yaml
    python main.py

done
