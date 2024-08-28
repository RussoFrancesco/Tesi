import csv
import os
import time

def write_on_file(filename, i, num_tokens,inference_time, cpu_usage_before, cpu_usage_after, memory_usage_before, memory_usage_after, score, bleu, hallucination):
    headers = False
    if not os.path.exists(filename):
        headers = True
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if headers:
            writer.writerow(["Input Text Index", "Input Tokens","Tempo di inferenza", "Uso CPU prima", "Uso CPU dopo", "Uso memoria prima", "Uso memoria dopo", "Perplexity", "Bleu", "Hallucination"])
            headers = False  
        writer.writerow([i, num_tokens,inference_time, cpu_usage_before, cpu_usage_after, memory_usage_before, memory_usage_after, score, bleu, hallucination])

def end_testing(filename):
    with open(filename, 'a', newline='') as f:
        current_time = time.localtime()
        formatted_time = time.strftime("%d-%m-%Y %H:%M:%S", current_time)
        f.write(f"Fine esecuzione: {formatted_time}")