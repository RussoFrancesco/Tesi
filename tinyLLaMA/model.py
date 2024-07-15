from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import time
import psutil
import torch
import csv
import os

model_path = "TinyLlama/TinyLlama_v1.1"


def calculate_perplexity(model, tokenizer, text):
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
    return perplexity


tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)


dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
texts = dataset['text']  


headers = False
filename = 'tinyLLaMA/tinyLLaMA.csv'
if not os.path.exists(filename):
    headers = True


for i, input_text in enumerate(texts):
    if len(input_text.strip()) == 0:
        continue  


    cpu_usage_before = psutil.cpu_percent(interval=1)
    memory_usage_before = psutil.virtual_memory().used


    start_time = time.time()
    generated_text = generator(input_text, max_new_tokens=100, num_return_sequences=1)[0]['generated_text']
    end_time = time.time()
    inference_time = end_time - start_time


    score = calculate_perplexity(model, tokenizer, generated_text)


    cpu_usage_after = psutil.cpu_percent(interval=1)
    memory_usage_after = psutil.virtual_memory().used


    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if headers:
            writer.writerow(["Input Text Index", "Tempo di inferenza", "Uso CPU prima", "Uso CPU dopo", "Uso memoria prima", "Uso memoria dopo", "Score"])
            headers = False  
        writer.writerow([i, inference_time, cpu_usage_before, cpu_usage_after, memory_usage_before, memory_usage_after, score])

