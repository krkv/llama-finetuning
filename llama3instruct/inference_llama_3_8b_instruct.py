from peft import PeftModel
from accelerate.utils import is_xpu_available
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import os

####################
# Config variables #
####################
seed: int = 42 # Seed value for reproducibility
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
peft_model: str = os.path.join(os.getcwd(), "llama-finetuning", "finetuned", "Llama-3-8B-Instruct", "01")
quantization: bool = True
max_new_tokens = 100 #The maximum numbers of tokens to generate
use_fast_kernels: bool = True # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels.
max_padding_length: int = None
do_sample: bool = True
min_length: int = None
use_cache: bool  = True # [optional] Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
repetition_penalty: float = 1.0 # The parameter for repetition penalty. 1.0 means no penalty.
length_penalty: int = 1  # [optional] Exponential penalty to the length that is used with beam-based generation.
temperature: float = 1.0 # [optional] The value used to modulate the next token probabilities.
top_p: float = 1.0 # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
top_k: int = 50 # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.

# define the current user input
question = "What are the most important features for instances with outdoor temperature higher than 28 degrees?"

# define tokens specific to Llama3-instruct prompting   
start_header_id = "<|start_header_id|>"
end_header_id = "<|end_header_id|>"
begin_of_text = "<|begin_of_text|>"
eot_id = "<|eot_id|>"

# read the system prompt from a file
with open(os.path.join(os.getcwd(), 'llama3instruct', 'system_prompt.txt'), 'r') as file:
    system_prompt = file.read()


def load_model(model_name, quantization, use_fast_kernels):
    print(f"use_fast_kernels{use_fast_kernels}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa" if use_fast_kernels else None,
    )
    return model


def load_peft_model(model, peft_model):
    peft_model = PeftModel.from_pretrained(model, peft_model)
    return peft_model


def inference(user_input):
    if len(user_input) < 1:
        raise RuntimeError("User input is empty")

    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(seed)
    else:
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
    model = load_model(model_name, quantization, use_fast_kernels)
    model = load_peft_model(model, peft_model)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # compose a prompt in the specific Llama3-Instruct format
    user_prompt = (
        f"{begin_of_text}{start_header_id}system{end_header_id}\n{system_prompt}{eot_id}{start_header_id}user{end_header_id}\n{user_input}{eot_id}{start_header_id}assistant{end_header_id}"
    )

    batch = tokenizer(user_prompt, padding='max_length', truncation=True, max_length=max_padding_length, return_tensors="pt")
    
    if is_xpu_available():
        batch = {k: v.to("xpu") for k, v in batch.items()}
    else:
        batch = {k: v.to("cuda") for k, v in batch.items()}

    start = time.perf_counter()
    
    with torch.no_grad():
        outputs = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            min_length=min_length,
            use_cache=use_cache,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty
        )
   
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    e2e_inference_time = (time.perf_counter()-start)*1000
    
    print(f"the inference time is {e2e_inference_time} ms")

    return output_text, e2e_inference_time

result = inference()
print(result)