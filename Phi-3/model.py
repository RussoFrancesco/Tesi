from transformers import pipeline, Phi3ForCausalLM, AutoTokenizer
from datasets import load_dataset
import time
import psutil
import csv
import os
from evaluate import load

model_path = "microsoft/Phi-3-mini-4k-instruct"


perplexity_metric = load("perplexity", module_type="metric")

def calculate_perplexity(text):
    results = perplexity_metric.compute(predictions=[text], model_id=model_path)
    print(results)
    return results['mean_perplexity']


model = Phi3ForCausalLM.from_pretrained(model_path) 
tokenizer = AutoTokenizer.from_pretrained(model_path)

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)


dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
texts = dataset['text']  


headers = False
filename = 'Phi-3/phi-3.csv'
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

    print(generated_text)
    score = calculate_perplexity(generated_text)


    cpu_usage_after = psutil.cpu_percent(interval=1)
    memory_usage_after = psutil.virtual_memory().used


    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if headers:
            writer.writerow(["Input Text Index", "Tempo di inferenza", "Uso CPU prima", "Uso CPU dopo", "Uso memoria prima", "Uso memoria dopo", "Score"])
            headers = False  
        writer.writerow([i, inference_time, cpu_usage_before, cpu_usage_after, memory_usage_before, memory_usage_after, score])

