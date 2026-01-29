
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock sentencepiece to avoid dependencies
sys.modules["sentencepiece"] = MagicMock()

# Add gguf-py to path
sys.path.insert(0, str(Path("/home/david/dev/BitNet-On-RPI4-Trixie/3rdparty/llama.cpp/gguf-py")))
from gguf import GGUFReader

model_path = "/home/david/dev/BitNet-On-RPI4-Trixie/models/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf"
reader = GGUFReader(model_path)

target = "blk.0.attn_q.weight"
for i, tensor in enumerate(reader.tensors):
    if tensor.name == target:
        print(f"Tensor: {tensor.name}")
        print(f"Data Offset: {tensor.data_offset}")
        print(f"n_bytes (GGUFReader): {tensor.n_bytes}")
        print(f"Expected n_bytes (ne[0]*ne[1]//4): {tensor.shape[0] * tensor.shape[1] // 4}")
        
        if i + 1 < len(reader.tensors):
            next_tensor = reader.tensors[i+1]
            print(f"Next Tensor: {next_tensor.name}")
            print(f"Next Data Offset: {next_tensor.data_offset}")
            gap = next_tensor.data_offset - tensor.data_offset
            print(f"Physical gap in file: {gap}")
            print(f"Difference (gap - n_bytes): {gap - tensor.n_bytes}")
        break
