from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import time
import psutil
import torch
import csv
import os
from huggingface_hub import login
from evaluate import load


login(token='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')

model_path = "google/gemma-2b"

perplexity_metric = load("perplexity", module_type="metric")

def calculate_perplexity(text):
    results = perplexity_metric.compute(predictions=[text], model_id=model_path)
    print(results)
    return results['mean_perplexity']

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)


dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
texts = dataset['text']  


headers = False
filename = 'Gemma/Gemma-2b-1.csv'
if not os.path.exists(filename):
    headers = True


for i, input_text in enumerate(texts):
    if len(input_text.strip()) == 0:
        continue  


    cpu_usage_before = psutil.cpu_percent(interval=1)
    memory_usage_before = psutil.virtual_memory().used

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    start_time = time.time()
    generation_output = model.generate(input_ids=input_ids, max_new_tokens=32)
    end_time = time.time()
    inference_time = end_time - start_time

    text_result = tokenizer.decode(generation_output[0])
    print(text_result)
    score = calculate_perplexity(text_result)


    cpu_usage_after = psutil.cpu_percent(interval=1)
    memory_usage_after = psutil.virtual_memory().used


    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if headers:
            writer.writerow(["Input Text Index", "Tempo di inferenza", "Uso CPU prima", "Uso CPU dopo", "Uso memoria prima", "Uso memoria dopo", "Score"])
            headers = False  
        writer.writerow([i, inference_time, cpu_usage_before, cpu_usage_after, memory_usage_before, memory_usage_after, score])


