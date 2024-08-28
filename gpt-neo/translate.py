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

# Funzione per ottenere l'utilizzo della CPU e della memoria
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

    return float(cpu_usage), float(mem_usage)

# Impostazione del percorso del progetto
percorso_progetto = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if percorso_progetto not in sys.path:
    sys.path.append(percorso_progetto)

from hallucination import calculate_hallucination
from write_on_file import write_on_file

model_params = sys.argv[1]
model_path = f'EleutherAI/gpt-neo-{model_params}'
filename = f'gpt-neo-{model_params}-mac.csv'
process = psutil.Process(os.getpid())

nltk.download('punkt')

# Caricamento delle metriche
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

# Caricamento del tokenizer e del modello GPT-Neo
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPTNeoForCausalLM.from_pretrained(model_path)

# Inizializzazione della pipeline di generazione testo
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Caricamento del dataset Fleurs
dataset = load_dataset("facebook/flores", "ita_Latn-eng_Latn",split='dev', trust_remote_code=True)
source_sentence = "sentence_ita_Latn"
target_sentence = "sentence_eng_Latn"


for i in range(len(dataset)):
    if i >= 100:
        break

    src_text=dataset[i][source_sentence]
    tgt_text=dataset[i][target_sentence]
    # Monitoraggio utilizzo risorse
    cpu_usage_before = psutil.cpu_percent(interval=0.1)
    memory_usage_before = psutil.virtual_memory().percent
    num_tokens = tokenizer(src_text, return_tensors="pt").input_ids.shape[-1]
    print(tgt_text+"\n")

    # Generazione della traduzione con GPT-Neo
    prompt = f"Traduci il seguente testo dall'italiano all'inglese:\nItaliano: {src_text}\nInglese:"
    print(prompt)
    start_time = time.time()
    generated_text = generator(prompt, max_new_tokens=400, num_return_sequences=1)[0]['generated_text']
    end_time = time.time()

    print(generated_text)

    # Estrazione della parte tradotta dal testo generato
    translated_text = generated_text.split("Translation:")[1].strip() if "Translation:" in generated_text else generated_text

    cpu_usage_after = psutil.cpu_percent(interval=0.1)
    memory_usage_after = psutil.virtual_memory().percent
    inference_time = end_time - start_time

    # Calcolo delle metriche
    perplexity = calculate_perplexity(translated_text)
    bleu = calculate_bleu(target_sentence, translated_text)
    #hallucination = calculate_hallucination(input_text, translated_text)

    # Scrittura dei risultati nel file CSV
    write_on_file(filename, i, num_tokens, inference_time, cpu_usage_before, cpu_usage_after, memory_usage_before, memory_usage_after, perplexity, bleu, hallucination=None)
