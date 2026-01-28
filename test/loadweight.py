#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from safetensors import safe_open

# Configuration
file_path = "../models/bitnet-b1.58-2B-4T-bf16/model.safetensors"
tensor_name = "model.layers.0.self_attn.q_proj.weight"

# Load the safetensors file
print(f"Loading from {file_path}...")
with safe_open(file_path, framework="pt", device="cpu") as f:
    # List available tensors
    available_tensors = f.keys()
    print(f"Available tensors ({len(available_tensors)} total):")
    for i, key in enumerate(list(available_tensors)[:10]):
        print(f"  {i}: {key}")
    if len(available_tensors) > 10:
        print(f"  ... and {len(available_tensors) - 10} more")
    
    # Load the specific tensor
    if tensor_name not in available_tensors:
        print(f"\nError: Tensor '{tensor_name}' not found!")
        print("Available tensors:")
        for key in available_tensors:
            print(f"  - {key}")
        exit(1)
    
    weights = f.get_tensor(tensor_name)
    print(f"\nLoaded tensor '{tensor_name}'")
    print(f"Shape: {weights.shape}")
    print(f"Dtype: {weights.dtype}")
    print(f"Total elements: {weights.numel()}")
    
    # Convert to float32 for numpy compatibility
    weights = weights.float()
    
    # Convert to numpy for analysis
    weights_np = weights.cpu().numpy().flatten()
    
    print(f"\nWeight statistics:")
    print(f"  Min: {weights_np.min():.6f}")
    print(f"  Max: {weights_np.max():.6f}")
    print(f"  Mean: {weights_np.mean():.6f}")
    print(f"  Std: {weights_np.std():.6f}")
    
    # Count values near {-1, 0, 1}
    near_neg1 = np.sum(np.abs(weights_np - (-1.0)) < 0.1)
    near_zero = np.sum(np.abs(weights_np - 0.0) < 0.1)
    near_pos1 = np.sum(np.abs(weights_np - 1.0) < 0.1)
    
    print(f"\nClustering analysis (within ±0.1):")
    print(f"  Near -1: {near_neg1} ({100*near_neg1/len(weights_np):.2f}%)")
    print(f"  Near 0: {near_zero} ({100*near_zero/len(weights_np):.2f}%)")
    print(f"  Near +1: {near_pos1} ({100*near_pos1/len(weights_np):.2f}%)")
    
    # Plot histogram
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(weights_np, bins=100, edgecolor='black', alpha=0.7)
    plt.axvline(-1, color='r', linestyle='--', label='-1')
    plt.axvline(0, color='g', linestyle='--', label='0')
    plt.axvline(1, color='b', linestyle='--', label='+1')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.title(f'Weight Distribution Histogram\n{tensor_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Zoomed in histogram around ternary values
    plt.subplot(1, 2, 2)
    weights_zoomed = weights_np[(weights_np >= -1.5) & (weights_np <= 1.5)]
    plt.hist(weights_zoomed, bins=200, edgecolor='black', alpha=0.7)
    plt.axvline(-1, color='r', linestyle='--', linewidth=2, label='-1')
    plt.axvline(0, color='g', linestyle='--', linewidth=2, label='0')
    plt.axvline(1, color='b', linestyle='--', linewidth=2, label='+1')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.title('Zoomed: [-1.5, 1.5] Range')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('weight_histogram.png', dpi=150)
    print(f"\nHistogram saved to weight_histogram.png")
    plt.show()
