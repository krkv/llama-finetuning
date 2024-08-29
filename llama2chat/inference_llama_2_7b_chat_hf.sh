#!/bin/bash

python ~/ttm/llama-recipes/recipes/inference/local_inference/inference.py --model_name meta-llama/Llama-2-7b-chat-hf --peft_model ~/ttm/llama-finetuning/finetuned/llama-2-7b-chat-hf/03 --prompt_file ~/ttm/llama-finetuning/llama2chat/prompt.txt