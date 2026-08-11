from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

text = "Hello, I am learning Large Language Models."

tokens = tokenizer.tokenize(text)
ids = tokenizer.encode(text)

print("Original:")
print(text)

print("\nTokens:")
print(tokens)

print("\nToken IDs:")
print(ids)

print("\nDecoded:")
print(tokenizer.decode(ids))