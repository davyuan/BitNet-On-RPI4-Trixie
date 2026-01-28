import torch
import numpy as np
import os

def quantize_with_gamma(weights_tensor, gamma, epsilon=1e-7):
    """
    Apply the exact quantization logic from C++:
    - normalized = weight / (gamma + epsilon)
    - rounded = round(normalized)
    - clipped to [-1, 1]
    """
    normalized = weights_tensor / (gamma + epsilon)
    rounded = torch.round(normalized)
    quantized = torch.clamp(rounded, -1.0, 1.0)
    return quantized

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

    # 2. Calculate the Global Scaler (absmean)
    beta = torch.mean(torch.abs(weights_tensor)).item()
    print(f"Calculated Global Beta (Absmean): {beta:.6f}")

    best_similarity = -1.0
    best_gamma = 0.0

    # 3. Sweep gamma values to find optimal
    # Gamma typically ranges around the mean absolute value
    gamma_range = np.linspace(beta * 0.1, beta * 5.0, 191)
    
    print("Sweeping gamma values...")
    similarities = []
    gammas = []
    
    for gamma in gamma_range:
        # Quantize using exact C++ logic
        quantized = quantize_with_gamma(weights_tensor, gamma)
        
        # Dequantize: multiply by gamma
        weight_hat = quantized * gamma
        
        # Calculate Cosine Similarity
        sim = torch.nn.functional.cosine_similarity(
            weights_tensor.view(-1), 
            weight_hat.view(-1), 
            dim=0
        ).item()
        
        similarities.append(sim)
        gammas.append(gamma)
        
        if sim > best_similarity:
            best_similarity = sim
            best_gamma = gamma

    print("-" * 60)
    print(f"RESULTS FOR {file_path}")
    print(f"Peak Cosine Similarity: {best_similarity:.6f}")
    print(f"Optimal Gamma: {best_gamma:.6f}")
    print(f"Global Beta (absmean): {beta:.6f}")
    print(f"Gamma / Beta ratio: {best_gamma / beta:.4f}")
    print("-" * 60)
    
    # Apply optimal quantization and show statistics
    print("\nApplying optimal quantization...")
    quantized_optimal = quantize_with_gamma(weights_tensor, best_gamma)
    weight_hat_optimal = quantized_optimal * best_gamma
    
    print(f"\nQuantization Statistics:")
    print(f"  Quantized values: {torch.unique(quantized_optimal).tolist()}")
    print(f"  Count at -1: {(quantized_optimal == -1).sum().item()}")
    print(f"  Count at 0: {(quantized_optimal == 0).sum().item()}")
    print(f"  Count at +1: {(quantized_optimal == 1).sum().item()}")
    
    # Calculate reconstruction error
    error = (weights_tensor - weight_hat_optimal).abs()
    print(f"\nReconstruction Error:")
    print(f"  Mean error: {error.mean().item():.6f}")
    print(f"  Max error: {error.max().item():.6f}")
    print(f"  RMSE: {torch.sqrt(torch.mean(error**2)).item():.6f}")
    
    # Plot results
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 5))
        
        # Similarity vs gamma
        plt.subplot(1, 2, 1)
        plt.plot(gammas, similarities, 'b-', linewidth=1)
        plt.axvline(best_gamma, color='r', linestyle='--', label=f'Optimal: {best_gamma:.6f}')
        plt.axvline(beta, color='g', linestyle='--', label=f'Beta: {beta:.6f}')
        plt.xlabel('Gamma')
        plt.ylabel('Cosine Similarity')
        plt.title('Cosine Similarity vs Gamma')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Original vs reconstructed
        plt.subplot(1, 2, 2)
        weights_flat = weights_tensor.view(-1).numpy()
        reconstructed_flat = weight_hat_optimal.view(-1).detach().numpy()
        plt.scatter(weights_flat[::100], reconstructed_flat[::100], alpha=0.5, s=1)
        min_val = min(weights_flat.min(), reconstructed_flat.min())
        max_val = max(weights_flat.max(), reconstructed_flat.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
        plt.xlabel('Original Weights')
        plt.ylabel('Reconstructed Weights')
        plt.title('Original vs Reconstructed')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('gamma_optimization.png', dpi=150)
        print(f"\nPlot saved to gamma_optimization.png")
    except ImportError:
        print("Matplotlib not available, skipping plot")

# Run the analysis
analyze_q_proj_quantization("q_proj_weights.bin")