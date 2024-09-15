#!/bin/bash

#SBATCH --job-name="ttm"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15000MB
#SBATCH --time=08:00:00

module load any/python/3.9.9
source ~/ttm/llama-finetuning/venv-llama-finetuning/bin/activate

srun python -m llama_recipes.finetuning --dataset "custom_dataset" --custom_dataset.file "../llama-dataset/llama3/energy_dataset.py" --use_peft --peft_method lora --quantization --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --output_dir "finetuned/Llama-3_1-8B-Instruct/01" --use_wandb


