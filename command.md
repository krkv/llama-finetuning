python llama-recipes/recipes/finetuning/finetuning.py -use_peft --peft_method lora --quantization --model_name meta-llama/Llama-2-7b-chat-hf --output_dir ~/ttm/llama/models/output --use_wandb

python llama-recipes/recipes/finetuning/finetuning.py --dataset "custom_dataset" --custom_dataset.file ~/ttm/llama/energy-dataset/energy_dataset.py -use_peft --peft_method lora --quantization --model_name meta-llama/Llama-2-7b-chat-hf --output_dir ~/ttm/llama/models/output

Inference:

python ~/ttm/llama-recipes/recipes/inference/local_inference/inference.py --model_name meta-llama/Llama-2-7b-chat-hf --peft_model ~/ttm/llama/models/llama_2_7b_chat_hf_02 --prompt_file ~/ttm/llama/prompt.txt