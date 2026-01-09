#!/usr/bin/env python3
import json
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python inspect_model.py <model_path>")
    sys.exit(1)

model_path = Path(sys.argv[1])
config_file = model_path / "config.json"

if not config_file.exists():
    print(f"Config file not found: {config_file}")
    sys.exit(1)

with open(config_file) as f:
    config = json.load(f)

# Print relevant attention dimensions
print("Model Config - Attention Dimensions:")
print(f"  hidden_size (n_embd): {config.get('hidden_size', 'N/A')}")
print(f"  num_attention_heads: {config.get('num_attention_heads', 'N/A')}")
print(f"  num_key_value_heads: {config.get('num_key_value_heads', 'N/A')}")

hidden_size = config.get('hidden_size', 0)
num_heads = config.get('num_attention_heads', 0)

if hidden_size and num_heads:
    head_dim = hidden_size // num_heads
    print(f"\nCalculated head_dim: {head_dim}")
    print(f"Expected attn_q.weight shape: ({hidden_size}, {hidden_size})")
    print(f"Expected attn_k.weight shape: ({hidden_size}, {hidden_size // num_heads * num_heads})")

print("\nFull config:")
print(json.dumps(config, indent=2))
