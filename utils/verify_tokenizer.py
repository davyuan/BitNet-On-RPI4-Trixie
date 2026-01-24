from gguf import GGUFReader

model_path = "models/bitnet-b1.58-2B-4T-bf16/ggml-model-tl1.gguf"
reader = GGUFReader(model_path)

# Find the specific token index
target_index = 1567
token_key = "tokenizer.ggml.tokens"

for field in reader.fields.values():
    if field.name == token_key:
        if target_index < len(field.parts):
            # field.parts contains the raw bytes of the tokens
            token_bytes = field.parts[target_index]
            print(f"Token at [{target_index}]: {token_bytes.tobytes().decode('utf-8', errors='replace')}")
        else:
            print(f"Index {target_index} is out of bounds for vocab size {len(field.parts)}")
        break
else:
    print(f"Key '{token_key}' not found in metadata.")