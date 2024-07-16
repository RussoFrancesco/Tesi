from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset
import time
import psutil
import torch
import csv
import os
from evaluate import load

model_path = 'mtgv/MobileLLaMA-1.4B-Base'

perplexity_metric = load("perplexity", module_type="metric")

def calculate_perplexity(text):
    results = perplexity_metric.compute(predictions=[text], model_id=model_path)
    print(results)
    return results['mean_perplexity']


tokenizer = LlamaTokenizer.from_pretrained(model_path, legacy=False)
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16
)

dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
texts = dataset['text']  


headers = False
filename = 'mobileLLaMA/mobileLLaMA-2.csv'
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

    text_results = tokenizer.decode(generation_output[0], skip_special_tokens=True)
    print(text_results)
    score = calculate_perplexity(text_results)


    cpu_usage_after = psutil.cpu_percent(interval=1)
    memory_usage_after = psutil.virtual_memory().used


    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if headers:
            writer.writerow(["Input Text Index", "Tempo di inferenza", "Uso CPU prima", "Uso CPU dopo", "Uso memoria prima", "Uso memoria dopo", "Score"])
            headers = False  
        writer.writerow([i, inference_time, cpu_usage_before, cpu_usage_after, memory_usage_before, memory_usage_after, score])


