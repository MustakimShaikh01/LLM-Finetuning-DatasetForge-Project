from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

embeddings = model.get_input_embeddings()

print(embeddings)

print("\nEmbedding Matrix Shape:")
print(embeddings.weight.shape)