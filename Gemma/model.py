from transformers import pipeline, GemmaTokenizer, GemmaForCausalLM
from datasets import load_dataset
import time
import psutil
import torch
import os
from evaluate import load
from huggingface_hub import login
import nltk
from nltk.translate.bleu_score import sentence_bleu
import sys

percorso_progetto = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


if percorso_progetto not in sys.path:
    sys.path.append(percorso_progetto)

from hallucination import calculate_hallucination
from write_on_file import write_on_file


login(token='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')

model_path = "google/gemma-2b"
filename = 'Gemma/Gemma-2b-1.csv'

perplexity_metric = load("perplexity", module_type="metric")

def calculate_perplexity(text):
    results = perplexity_metric.compute(predictions=[text], model_id=model_path)
    print(results)
    return results['mean_perplexity']

def calculate_perplexity(text):
    results = perplexity_metric.compute(predictions=[text], model_id=model_path)
    print(results)
    return results['mean_perplexity']

def calculate_bleu(reference, text):
    reference_tokens = nltk.word_tokenize(reference)
    text_tokens = nltk.word_tokenize(text)
    bleu = sentence_bleu([reference_tokens], text_tokens)
    return bleu


tokenizer = GemmaTokenizer.from_pretrained(model_path)
model = GemmaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)


dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
texts = dataset['text']  


for i, input_text in enumerate(texts):

    cpu_usage_before = psutil.cpu_percent(interval=1)
    memory_usage_before = psutil.virtual_memory().used

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    num_tokens = input_ids.shape[-1]

    start_time = time.time()
    generation_output = model.generate(input_ids=input_ids, max_new_tokens=32)
    end_time = time.time()
    inference_time = end_time - start_time

    cpu_usage_after = psutil.cpu_percent(interval=1)
    memory_usage_after = psutil.virtual_memory().used


    text_result = tokenizer.decode(generation_output[0])
    score = calculate_perplexity(text_result)
    bleu = calculate_bleu(input_text, text_result)
    hallucination = calculate_hallucination(input_text, text_result)

    write_on_file(filename, i, num_tokens, inference_time, cpu_usage_before, cpu_usage_after, memory_usage_before, memory_usage_after, score, bleu, hallucination)


