#!/usr/bin/env python3
import numpy as np
import sys
from pathlib import Path

# Add gguf-py to path
gguf_path = Path(__file__).parent.parent / "3rdparty/llama.cpp/gguf-py"
sys.path.insert(0, str(gguf_path))

from gguf import GGUFReader

# Configuration
file_path = "./models/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf"
tensor_name = "blk.5.ffn_down.weight"

def unpack_ternary_2bit(byte_val):
    """
    Unpack 4 ternary weights from a single uint8_t.
    Big endian: bits 7-6, 5-4, 3-2, 1-0
    Mapping: 0b00 = -1, 0b01 = 0, 0b10 = 1, 0b11 = undefined
    """
    weights = []
    for i in range(4):
        # Extract 2 bits: (byte_val >> (6 - i*2)) & 0x03
        shift = 6 - i * 2
        two_bits = (byte_val >> shift) & 0x03
        
        # Map 2-bit value to ternary
        if two_bits == 0b00:
            weight = -1
        elif two_bits == 0b01:
            weight = 0
        elif two_bits == 0b10:
            weight = 1
        else:  # 0b11
            weight = 0  # undefined, default to 0
            print(f"Warning: encountered undefined 2-bit value {two_bits:#04b}, defaulting to 0")
        
        weights.append(weight)
    
    return weights

# Load the GGUF file
print(f"Loading from {file_path}...")
reader = GGUFReader(file_path, 'r')

# Find and load the tensor
tensor_found = None
for tensor in reader.tensors:
    if tensor.name == tensor_name:
        tensor_found = tensor
        break

if tensor_found is None:
    print(f"Tensor '{tensor_name}' not found in GGUF file!")
    print(f"Available tensors: {[t.name for t in reader.tensors[:10]]}...")
    sys.exit(1)

print(f"Loaded tensor '{tensor_name}'")
print(f"Shape: {tensor_found.shape}")
print(f"Tensor type: {tensor_found.tensor_type}")

# Get the tensor data
weights_np = tensor_found.data
if weights_np is None:
    print("Error: Could not load tensor data")
    sys.exit(1)

# Ensure it's a numpy array
if not isinstance(weights_np, np.ndarray):
    weights_np = np.array(weights_np)

print(f"\n=== First 16 rows of weights (first 16 elements per row) ===\n")

# Print first 16 rows, first 16 columns
for row in range(min(16, weights_np.shape[0])):
    print(f"Row {row:2d}: ", end="")
    for col in range(min(16, weights_np.shape[1])):
        val = weights_np[row, col]
        print(f"{int(val):02x} ", end="")
    print()

print(f"\n=== If this were packed as ternary (2-bit per weight) ===\n")
print("Simulating packed format: 4 weights per uint8_t, big-endian 2-bit")
print("Mapping: 0b00=-1, 0b01=0, 0b10=1, 0b11=undefined\n")

# Initialize counters
neg1_count = 0
zero_count = 0
pos1_count = 0

for row in range(min(16, weights_np.shape[0])):
    print(f"Row {row:2d}: ", end="")
    for col in range(min(8, weights_np.shape[1])):
        # Unpack and display
        unpacked = unpack_ternary_2bit(weights_np[row, col].astype(np.uint8))
        print(f"[{unpacked[0]:2d},{unpacked[1]:2d},{unpacked[2]:2d},{unpacked[3]:2d}] ", end="")
    print()

# Count all weights across entire tensor
for row in range(weights_np.shape[0]):
    for col in range(weights_np.shape[1]):
        unpacked = unpack_ternary_2bit(weights_np[row, col].astype(np.uint8))
        for weight in unpacked:
            if weight == -1:
                neg1_count += 1
            elif weight == 0:
                zero_count += 1
            else:  # weight == 1
                pos1_count += 1

print(f"\n=== Statistics ===")
size = neg1_count + zero_count + pos1_count
print(f"Total ternary weights: {size}")
print(f"  Count at -1: {neg1_count} ({100*neg1_count/size:.2f}%)")
print(f"  Count at 0: {zero_count} ({100*zero_count/size:.2f}%)")
print(f"  Count at +1: {pos1_count} ({100*pos1_count/size:.2f}%)")