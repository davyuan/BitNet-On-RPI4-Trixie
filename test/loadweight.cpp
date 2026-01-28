#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>

#include "safetensors.h"

static float bf16_to_f32(uint16_t v) {
    uint32_t u = uint32_t(v) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

int main() {
    const char *file = "../models/bitnet-b1.58-2B-4T-bf16/model.safetensors";
    const std::string tensor_name = "model.layers.0.self_attn.q_proj.weight";
    const char *save_to_file = "q_proj_weights.bin";

    safetensors::safetensors_t st;
    std::string err, warn;
    
    // Load safetensors from file
    if (!safetensors::load_from_file(file, &st, &warn, &err)) {
        std::cerr << "Load error: " << err << "\n";
        if (!warn.empty()) {
            std::cerr << "Warning: " << warn << "\n";
        }
        return 1;
    }

    // Find the tensor
    if (!st.tensors.count(tensor_name)) {
        std::cerr << "Tensor '" << tensor_name << "' not found\n";
        std::cerr << "Available tensors:\n";
        for (const auto& key : st.tensors.keys()) {
            std::cerr << "  - " << key << "\n";
        }
        return 1;
    }

    safetensors::tensor_t tensor;
    if (!st.tensors.at(tensor_name, &tensor)) {
        std::cerr << "Failed to get tensor\n";
        return 1;
    }
    
    // Check dtype
    if (tensor.dtype != safetensors::kBFLOAT16) {
        std::cerr << "Expected BF16, but got dtype: " << tensor.dtype << "\n";
        return 1;
    }

    // Calculate total elements from shape
    size_t numel = 1;
    for (auto d : tensor.shape) numel *= d;

    std::cout << "Tensor shape: [";
    for (size_t i = 0; i < tensor.shape.size(); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << tensor.shape[i];
    }
    std::cout << "]\n";
    std::cout << "Total elements: " << numel << "\n";
    std::cout << "Data offsets: [" << tensor.data_offsets[0] << ", " << tensor.data_offsets[1] << "]\n";
    std::cout << "Storage size: " << st.storage.size() << "\n";
    std::cout << "Databuffer addr: " << (void*)st.databuffer_addr << "\n";
    std::cout << "Databuffer size: " << st.databuffer_size << "\n";

    // Get pointer to BF16 data - use storage if databuffer_addr is not available
    const uint8_t *data_ptr;
    if (!st.storage.empty()) {
        // Data is in storage (copied mode)
        data_ptr = st.storage.data() + tensor.data_offsets[0];
        std::cout << "Using storage buffer\n";
    } else {
        // Data is in databuffer (mmap mode)
        data_ptr = st.databuffer_addr + tensor.data_offsets[0];
        std::cout << "Using databuffer\n";
    }
    
    const uint16_t *bf16_data = reinterpret_cast<const uint16_t *>(data_ptr);

    // Convert BF16 to F32
    std::vector<float> weights(numel);
    for (size_t i = 0; i < numel; i++) {
        weights[i] = bf16_to_f32(bf16_data[i]);
    }

    std::cout << "Loaded " << numel << " weights\n";
    std::cout << "First 10 values: ";
    for (size_t i = 0; i < std::min(size_t(10), numel); i++) {
        std::cout << weights[i] << " ";
    }
    std::cout << "\n";
    
    // Save weights to binary file
    std::ofstream outfile(save_to_file, std::ios::binary);
    if (!outfile) {
        std::cerr << "Failed to open file for writing: " << save_to_file << "\n";
        return 1;
    }
    
    outfile.write(reinterpret_cast<const char*>(weights.data()), weights.size() * sizeof(float));
    if (!outfile) {
        std::cerr << "Failed to write to file: " << save_to_file << "\n";
        return 1;
    }
    
    outfile.close();
    std::cout << "Saved " << numel << " weights to " << save_to_file << " (" 
              << (numel * sizeof(float) / 1024 / 1024) << " MB)\n";
    
    return 0;
}
