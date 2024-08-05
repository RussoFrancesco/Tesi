from transformers import AutoModelForCausalLM
from optimum.quanto import QuantizedModelForCausalLM, qint4

# Carica il modello
model = AutoModelForCausalLM.from_pretrained('mtgv/MobileLLaMA-1.4B-Base')

# Stampa la memoria occupata dal modello originale
original_memory_footprint = model.get_memory_footprint() / 1e6
print(f"Original model memory footprint: {original_memory_footprint:.2f} MB")

# Quantizza il modello
qmodel = QuantizedModelForCausalLM.quantize(model, weights=qint4, exclude='lm_head')

# Salva il modello quantizzato
qmodel.save_pretrained('./MobileLLaMA-1.4B-Base-quantized')

# Carica il modello quantizzato per verificarlo
quantized_model = QuantizedModelForCausalLM.from_pretrained('./MobileLLaMA-1.4B-Base-quantized')

# Stampa la memoria occupata dal modello quantizzato
quantized_memory_footprint = quantized_model.get_memory_footprint() / 1e6
print(f"Quantized model memory footprint: {quantized_memory_footprint:.2f} MB")

# Confronta il numero di parametri e il tipo di dato di ogni parametro
for name, param in quantized_model.named_parameters():
    print(f"{name}: {param.dtype}")
