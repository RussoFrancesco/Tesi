from transformers import pipeline, GPTNeoForCausalLM, GPT2Tokenizer
from datasets import load_dataset
import time
import psutil
import os
import subprocess
import torch
from evaluate import load
import nltk
from nltk.translate.bleu_score import sentence_bleu
import sys
import threading

def getCPUuse():
    process = subprocess.Popen(['top', '-b', '-n', '1', '-p', str(os.getpid())], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    stdout = stdout.decode()
    cpu_usage = None
    mem_usage = None

    for line in stdout.splitlines():
        if 'pt_main' in line:
            values = line.split()
            
            cpu_usage = values[8]
            mem_usage = values[9]
            break 

    print(float(cpu_usage), float(mem_usage))


percorso_progetto = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


if percorso_progetto not in sys.path:
    sys.path.append(percorso_progetto)

from hallucination import calculate_hallucination
from write_on_file import write_on_file

model_params = sys.argv[1]

model_path = f'EleutherAI/gpt-neo-{model_params}'
filename = f'gpt-neo-{model_params}-raspberry.csv'
process = psutil.Process(os.getpid())

#nltk.download('punkt')


perplexity_metric = load("perplexity", module_type="metric")

def calculate_perplexity(text):
    results = perplexity_metric.compute(predictions=[text], model_id=model_path)
    print(results)
    return results['mean_perplexity']

def calculate_bleu(reference, text):
    reference_tokens = nltk.word_tokenize(reference)
    text_tokens = nltk.word_tokenize(text)
    bleu = sentence_bleu([reference_tokens], text_tokens)
    return bleu


tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPTNeoForCausalLM.from_pretrained(model_path)

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
texts = dataset['text']

thread = threading.Thread(target=getCPUuse)

for i, input_text in enumerate(texts):
    if i >= 100:
        break

    thread.start()
    #cpu_usage_before, memory_usage_before = getCPUuse()
    #cpu_usage_before /= 4
    print(psutil.cpu_times_percent(interval=0.1, percpu=False))
    cpu_usage_before = psutil.cpu_times_percent(interval=0.1, percpu=False)[0]
    memory_usage_before = psutil.virtual_memory().percent
    num_tokens = tokenizer(input_text, return_tensors="pt").input_ids.shape[-1]

    start_time = time.time()
    generated_text = generator(input_text, max_new_tokens=100, num_return_sequences=1)[0]['generated_text']
    end_time = time.time()

    cpu_usage_after = psutil.cpu_times_percent(interval=0.1, percpu=False)[0]
    memory_usage_after = psutil.virtual_memory().percent
    #cpu_usage_after, memory_usage_after = getCPUuse()
    #cpu_usage_after /= 4

    inference_time = end_time - start_time

    perplexity = calculate_perplexity(generated_text)
    bleu = calculate_bleu(input_text, generated_text)
    hallucination = calculate_hallucination(input_text, generated_text)


    write_on_file(filename, i, num_tokens, inference_time, cpu_usage_before, cpu_usage_after, memory_usage_before, memory_usage_after, perplexity, bleu, hallucination)
