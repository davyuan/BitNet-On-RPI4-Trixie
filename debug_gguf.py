#!/usr/bin/env python3
import sys
import os

# Add gguf-py to path
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '3rdparty/llama.cpp/gguf-py')))

from gguf import GGUFReader

if len(sys.argv) < 2:
    print("Usage: debug_gguf.py <gguf_file>")
    sys.exit(1)

reader = GGUFReader(sys.argv[1])

print(f"Total tensors: {len(reader.tensors)}")
print("\nTensor names:")
for i, tensor in enumerate(reader.tensors):
    print(f"  {i}: {tensor.name} (shape={tensor.shape}, type={tensor.tensor_type.name})")

print(f"\n\nLooking for _scale tensors:")
scale_tensors = [t for t in reader.tensors if "_scale" in t.name]
print(f"Found {len(scale_tensors)} scale tensors:")
for tensor in scale_tensors:
    print(f"  {tensor.name} (shape={tensor.shape}, type={tensor.tensor_type.name}, n_elements={tensor.n_elements})")
