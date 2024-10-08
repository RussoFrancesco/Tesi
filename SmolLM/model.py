from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import time
import psutil
import os
from evaluate import load
import nltk
from nltk.translate.bleu_score import sentence_bleu
import sys


percorso_progetto = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


if percorso_progetto not in sys.path:
    sys.path.append(percorso_progetto)

from hallucination import calculate_hallucination
from write_on_file import write_on_file, end_testing, start_testing

model_params = sys.argv[1]
start_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0

model_path = f'HuggingFaceTB/SmolLM-{model_params}'
filename = f'SmolLM-{model_params}-mac.csv'

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


tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
texts = dataset['text']

p = psutil.Process()
p.cpu_percent(interval=None)

start_testing(f"SmolLM-{model_params}-mac.txt")
for i, input_text in enumerate(texts[start_index:], start=start_index):
    if i >= start_index + 101:
        break
        
    cpu_usage_before = p.cpu_percent(interval=None)
    memory_usage_before = p.memory_percent()
    num_tokens = tokenizer(input_text, return_tensors="pt").input_ids.shape[-1]

    start_time = time.time()
    generated_text = generator(input_text, max_new_tokens=100, num_return_sequences=1)[0]['generated_text']
    end_time = time.time()

    cpu_usage_after = p.cpu_percent(interval=None)
    memory_usage_after = p.memory_percent()

    inference_time = end_time - start_time

    if cpu_usage_before > 100:
        cpu_usage_before = cpu_usage_before / os.cpu_count()
    
    if cpu_usage_after > 100:
        cpu_usage_after = cpu_usage_after / os.cpu_count()

    perplexity = calculate_perplexity(generated_text)
    bleu = calculate_bleu(input_text, generated_text)
    hallucination = calculate_hallucination(input_text, generated_text)


    write_on_file(filename, i, num_tokens, inference_time, cpu_usage_before, cpu_usage_after, memory_usage_before, memory_usage_after, perplexity, bleu, hallucination)

end_testing(f"SmolLM-{model_params}-mac.txt")
