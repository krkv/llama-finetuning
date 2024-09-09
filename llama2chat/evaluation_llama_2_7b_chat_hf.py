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
    

def calculate_accuracy(log_file):
    eval_log = pd.read_csv(log_file)
    eval_log = eval_log.drop_duplicates()
    correct_parses = 0
    correct_percent = 0
    expected_parses = eval_log['expected_parse'].values
    parsed_utterances = eval_log['parsed_utterance'].values
    log_size = len(expected_parses)
    for i in range(log_size):
        print("Expected: " + expected_parses[i], "- Generated: " + parsed_utterances[i])
        if expected_parses[i] == parsed_utterances[i]:
            correct_parses += 1
    print()
   
    if correct_parses > 0:
        correct_percent = round((correct_parses / log_size) * 100, 2)
        
    return str(correct_percent)


generate_results("../llm-evaluation/gold_parse_energy.csv")

acc = calculate_accuracy(EVALUATION_LOG_FILE)

print(f"Calculated accuracy: {acc}%")