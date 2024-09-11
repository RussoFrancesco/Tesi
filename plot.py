import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Funzione per caricare i dati e calcolare il tempo medio di inferenza
def load_and_calculate_mean_time(file_path):
    df = pd.read_csv(file_path).head(100)
    return np.mean(df['Perplexity'])

# File paths e labels
file_paths = [
    "SmolLM/SmolLM-1.7B-raspberry.csv",
    "Gemma/Gemma-2b-mac.csv"
]

labels = ["SmolLM-1.7B", "Gemma-2b"]

# Calcola i tempi medi per tutti i file
mean_times = [load_and_calculate_mean_time(fp) for fp in file_paths]

# Creazione del grafico
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()

bars = ax.bar(x, mean_times, width)

# Aggiungi etichette e titolo
ax.set_xlabel('Modello')
ax.set_ylabel('Perplexity media')
ax.set_title('Confronto perplexity media SmolLM e Gemma')
ax.set_xticks(x)
ax.set_xticklabels(labels)

# Aggiungi le etichette di valore alle barre
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 punti di offset verticale
                textcoords="offset points",
                ha='center', va='bottom')
    

# Mostra il grafico
plt.show()



