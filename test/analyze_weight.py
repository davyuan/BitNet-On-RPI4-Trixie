import torch
import numpy as np
import os

def analyze_q_proj_quantization(file_path, shape=(2560, 2560)):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    # 1. Load the raw float32 binary file
    print(f"Loading weights from {file_path}...")
    weights_np = np.fromfile(file_path, dtype=np.float32)
    
    if weights_np.size != shape[0] * shape[1]:
        print(f"Warning: File size {weights_np.size} does not match expected {shape[0]*shape[1]}")
        # Adjusting to actual size if necessary
        weights_tensor = torch.from_numpy(weights_np).to(torch.float32)
    else:
        weights_tensor = torch.from_numpy(weights_np).reshape(shape).to(torch.float32)

    # 2. Calculate the Global Scaler (Beta)
    beta = torch.mean(torch.abs(weights_tensor)).item()
    print(f"Calculated Global Beta (Absmean): {beta:.6f}")

    best_similarity = -1.0
    best_threshold_coeff = 0.0

    # 3. Sweep the threshold coefficient (t * beta)
    # We sweep around the 0.5 mark to see where the similarity peaks
    thresholds = np.linspace(0.1, 0.9, 81)
    
    print("Sweeping thresholds...")
    for t_coeff in thresholds:
        threshold_val = t_coeff * beta
        
        # Ternary Quantization
        weight_q = torch.zeros_like(weights_tensor)
        weight_q[weights_tensor > threshold_val] = 1
        weight_q[weights_tensor < -threshold_val] = -1
        
        # Dequantize (W_hat = W_q * beta)
        weight_hat = weight_q * beta
        
        # Calculate Cosine Similarity
        sim = torch.nn.functional.cosine_similarity(
            weights_tensor.view(-1), 
            weight_hat.view(-1), 
            dim=0
        ).item()
        
        if sim > best_similarity:
            best_similarity = sim
            best_threshold_coeff = t_coeff

    print("-" * 40)
    print(f"RESULTS FOR {file_path}")
    print(f"Peak Cosine Similarity: {best_similarity:.6f}")
    print(f"Optimal Threshold Coefficient: {best_threshold_coeff:.2f}")
    print(f"Actual Threshold Value: {best_threshold_coeff * beta:.6f}")
    print("-" * 40)
    
    # Check if the 0.5 coefficient (std::round) was the bottleneck
    if abs(best_threshold_coeff - 0.5) > 0.01:
        print(f"Insight: Your C++ round() logic is off by {best_threshold_coeff - 0.5:.2f}.")
    else:
        print("Insight: The standard 0.5 rounding threshold is mathematically optimal for this tensor.")

# Run the analysis
analyze_q_proj_quantization("q_proj_weights.bin")