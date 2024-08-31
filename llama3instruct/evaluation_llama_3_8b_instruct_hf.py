import csv
import inference_llama_3_8b_instruct
import pandas as pd

EVALUATION_LOG_FILE = 'llama3instruct/evaluation_log.csv'

def log_result(user_input, expected_parse, generated_parse, inference_time):
    with open(EVALUATION_LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([user_input, expected_parse, generated_parse, inference_time])

def generate_results(dataset_file):
    df = pd.read_csv(dataset_file)
    
    prompts = df['prompt'].values
    expected_parses = df['query'].values
    
    for i in range(len(prompts)):
        prompt = prompts[i]
        expected_parse = expected_parses[i]
        parsed_utterance, inference_time = inference_llama_3_8b_instruct.inference(prompt)
        log_result(prompt, expected_parse, parsed_utterance, inference_time)
    
generate_results("../llm-evaluation/gold_parse_energy.csv")