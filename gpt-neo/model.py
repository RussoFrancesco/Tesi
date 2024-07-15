from transformers import pipeline, GPTNeoForCausalLM, GPT2Tokenizer
from datasets import load_dataset
import time
import psutil
import csv
import os
from evaluate import load


model_path = 'EleutherAI/gpt-neo-125M'


perplexity_metric = load("perplexity", module_type="metric")

def calculate_perplexity(text):
    results = perplexity_metric.compute(predictions=[text], model_id=model_path)
    print(results)
    return results['mean_perplexity']

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPTNeoForCausalLM.from_pretrained(model_path)

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
texts = dataset['text']

headers = not os.path.exists('gpt-neo/gpt-neo-1.csv')

with open('gpt-neo/gpt-neo-1.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    if headers:
        writer.writerow(["Input Text Index", "Tempo di inferenza", "Uso CPU prima", "Uso CPU dopo", "Uso memoria prima", "Uso memoria dopo", "Score"])
    
    for i, input_text in enumerate(texts):
        if len(input_text.strip()) == 0:
            continue
        
        cpu_usage_before = psutil.cpu_percent(interval=1)
        memory_usage_before = psutil.virtual_memory().used

        start_time = time.time()
        generated_text = generator(input_text, max_new_tokens=100, num_return_sequences=1)[0]['generated_text']
        end_time = time.time()
        inference_time = end_time - start_time

        score = calculate_perplexity(generated_text)

        cpu_usage_after = psutil.cpu_percent(interval=1)
        memory_usage_after = psutil.virtual_memory().used

        writer.writerow([i, inference_time, cpu_usage_before, cpu_usage_after, memory_usage_before, memory_usage_after, score])
