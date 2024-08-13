from transformers import AutoModelForCausalLM
from optimum.quanto import QuantizedModelForCausalLM, qint4

model = AutoModelForCausalLM.from_pretrained('mtgv/MobileLLaMA-1.4B-Base')
qmodel = QuantizedModelForCausalLM.quantize(model, weights=qint4, exclude=['lm_head'])

qmodel.save_pretrained('./MobileLLaMA-1.4B-Base-quantized')

qmodel1 = QuantizedModelForCausalLM.from_pretrained('MobileLLaMA-1.4B-Base-quantized')

for name, param in qmodel1.named_parameters():
    print(f"{name}: {param.dtype}")
