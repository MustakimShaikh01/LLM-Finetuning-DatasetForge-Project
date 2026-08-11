from transformers import AutoTokenizer, AutoModelForCausalLM

import torch

model_name = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "Explain what a transformer is."

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))