#!/bin/bash

#SBATCH --job-name="ttm-llama2"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15000MB
#SBATCH --time=07:00:00

module load any/python/3.9.9
source ~/ttm/llama-finetuning/venv-llama-finetuning/bin/activate

srun python ~/ttm/llama-recipes/recipes/finetuning/finetuning.py --dataset "custom_dataset" --custom_dataset.file ~/ttm/llama-dataset/energy_dataset.py --use_peft --peft_method lora --quantization --model_name meta-llama/Llama-2-7b-chat-hf --output_dir ~/ttm/llama-finetuning/finetuned/llama-2-7b-chat-hf/03 --use_wandb


