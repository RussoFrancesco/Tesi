import pandas as pd
import os


def calculate_metrics(csv_file):
    df = pd.read_csv(csv_file)
    
    mean_inference_time = df['Tempo di inferenza'].mean()
    mean_cpu_increase = (df['Uso CPU dopo'] - df['Uso CPU prima']).mean()
    mean_memory_increase = (df['Uso memoria dopo'] - df['Uso memoria prima']).mean()
    mean_perplexity = df['Perplexity'].mean()
    hallucination_count = (df['Hallucination'] > 0.5).sum()
    
    return mean_inference_time, mean_cpu_increase, mean_memory_increase, mean_perplexity, hallucination_count

def write_results(output_txt_file, metrics):
    mean_inference_time, mean_cpu_increase, mean_memory_increase, mean_perplexity, hallucination_count = metrics
    with open(output_txt_file, 'w') as f:
        f.write(f'Tempo medio di inferenza: {mean_inference_time}\n')
        f.write(f'Aumento medio della CPU: {mean_cpu_increase}\n')
        f.write(f'Aumento medio della memoria usata: {mean_memory_increase}\n')
        f.write(f'Perplexity media: {mean_perplexity}\n')
        f.write(f'Numero di volte che l\'hallucination supera 0.5: {hallucination_count}\n')

main_dir = "."

for root, dirs, files in os.walk(main_dir):
    for file in files:
        if file.endswith(".csv"):
            csv_file = os.path.join(root, file)
            output_txt_file = os.path.join(root, file.replace('.csv', '-analisi.txt'))
            metrics = calculate_metrics(csv_file)
            write_results(output_txt_file, metrics)
