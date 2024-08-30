import csv
import inference_llama_2_7b_chat_hf
import pandas as pd

EVALUATION_LOG_FILE = 'llama2chat/evaluation_log.csv'

def log_result(user_input, expected_parse, generated_parse):
    with open(EVALUATION_LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([user_input, expected_parse, generated_parse])

def generate_results(dataset_file):
    df = pd.read_csv(dataset_file)
    
    prompts = df['prompt'].values
    expected_parses = df['query'].values
    
    for i in range(len(prompts)):
        prompt = prompts[i]
        expected_parse = expected_parses[i]
        parsed_utterance = inference_llama_2_7b_chat_hf.inference(prompt)
        log_result(prompt, expected_parse, parsed_utterance)
    
generate_results("../llm-evaluation/gold_parse_energy.csv")