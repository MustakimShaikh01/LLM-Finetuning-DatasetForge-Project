from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

print("\n========== MODEL INFO ==========")
print(model)

print("\n========== CONFIG ==========")
print(model.config)

print("\n========== PARAMETERS ==========")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")

print("\n========== TOKENIZER ==========")
print("Vocabulary Size:", tokenizer.vocab_size)
print("Model Max Length:", tokenizer.model_max_length)