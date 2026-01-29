#!/usr/bin/env python3
import sys
import numpy as np
import os
from pathlib import Path
from unittest.mock import MagicMock

# Mock sentencepiece to avoid dependencies if not installed
try:
    import sentencepiece
except ImportError:
    sys.modules["sentencepiece"] = MagicMock()

# Add gguf-py to path
gguf_path = Path(__file__).resolve().parent.parent / "3rdparty/llama.cpp/gguf-py"
sys.path.insert(0, str(gguf_path))

import gguf
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType, GGUFValueType

def i2_s_to_ternary(data):
    """
    Convert I2_S packed data to ternary weights.
    Each byte contains 4 weights (2 bits each).
    Mapping: 0b00 = -1, 0b01 = 0, 0b10 = 1, 0b11 = undefined (mapped to 0)
    """
    ternary_weights = []
    # Flatten if data is a 2D numpy array to ensure we iterate over bytes
    for byte in data.reshape(-1):
        for i in range(4):
            shift = 6 - i * 2
            two_bits = (byte >> shift) & 0x03
            if two_bits == 0:
                ternary_weights.append(-1)
            elif two_bits == 1:
                ternary_weights.append(0)
            elif two_bits == 2:
                ternary_weights.append(1)
            else:  # 0b11
                print(f"Warning: encountered undefined 2-bit value {two_bits:#04b}, defaulting to 0")
                ternary_weights.append(0)  # undefined, default to 0

    return np.array(ternary_weights, dtype=np.int8)

def encode_ternary(ternary_data, M, K):
    """
    Encode ternary weights into TL1 packed format.
    Each byte contains 4 weights (2 bits each).
    Mapping: -1 = 0b00, 0 = 0b01, 1 = 0b10
    """
    weight_num = M * K
    weight = np.reshape(ternary_data, (weight_num // 2, 2))
    hi_weight = np.multiply(np.split(weight, 2, axis=1)[0], 3)
    lo_weight = np.split(weight, 2, axis=1)[1]

    weight = np.reshape((hi_weight + lo_weight), weight_num // 2)

    weight = weight + 4
    weight = np.reshape(weight, (M, K // 2)).astype(np.uint8)

    return weight

def pack_tl1(weight_encoded, M, K):
    weight = weight_encoded.transpose(1, 0)  # K/2, M
    weight = weight.reshape((K // 2, M // 2, 2))
    weight_0 = weight[:, :, 0] << 4
    weight_1 = weight[:, :, 1]
    weight = weight_0 + weight_1
    return weight  #Shape (K/2, M/2)    

def transform_i2_s_to_tl1(data, M, K, scale):
    '''print(f"Debug: First 16 rows of raw I2_S data (16 bytes per row):")
    for r in range(min(16, data.shape[0])):
        row_str = " ".join(f"{b:02x}" for b in data[r, :min(16, data.shape[1])])
        print(f"Row {r:2d}: {row_str}")'''

    ternary_data = i2_s_to_ternary(data)
    '''print(f"Debug: First 16 rows of ternary weights (16 elements per row):")
    ternary_reshaped = ternary_data.reshape(M, K)
    for r in range(min(16, M)):
        row_str = " ".join(f"{val:2d}" for val in ternary_reshaped[r, :min(16, K)])
        print(f"Row {r:2d}: {row_str}")'''

    encoded = encode_ternary(ternary_data, M, K)
    '''print(f"Debug: First 16 rows of TL1 encoded weights (16 elements per row):")
    for r in range(min(16, encoded.shape[0])):
        row_str = " ".join(f"{val:02x}" for val in encoded[r, :min(16, encoded.shape[1])])
        print(f"Row {r:2d}: {row_str}")'''

    packed = pack_tl1(encoded, M, K)
    '''print(f"Debug: First 16 rows of packed TL1 data (16 bytes per row):")
    for r in range(min(16, packed.shape[0])):
        row_str = " ".join(f"{b:02x}" for b in packed[r, :min(16, packed.shape[1])])
        print(f"Row {r:2d}: {row_str}")'''

    return packed, scale

def convert_model(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found")
        return

    print(f"Loading I2_S model from: {input_path}")
    reader = GGUFReader(input_path)
    
    # Extract architecture for the writer
    arch_field = reader.get_field("general.architecture")
    if arch_field is None:
        print("Error: Could not find general.architecture in input file")
        return
    arch = str(bytes(arch_field.parts[-1]), encoding='utf-8')
    
    writer = GGUFWriter(output_path, arch)
    
    # 2) Copy metadata fields (KV pairs)
    print("Copying metadata...")
    for key, field in reader.fields.items():
        # GGUFWriter handles these automatically
        if key in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"):
            continue
            
        # 2) Special case for general_file_type (handle both dot and underscore styles)
        if key in ("general.file_type", "general_file_type"):
            original_val = field.parts[-1][0]
            print(f"Updating {key} from {original_val} to 38 (TL1)")
            writer.add_uint32(key, 38)
            continue

        vtype = field.types[0]
        
        # 3) Copying vocab and tokenizer (handled here as they are KV pairs)
        if vtype == GGUFValueType.STRING:
            val = str(bytes(field.parts[field.data[0]]), encoding='utf-8')
            writer.add_string(key, val)
        elif vtype == GGUFValueType.ARRAY:
            # Handle arrays (like tokens, merges, etc.)
            inner_type = field.types[1]
            items = []
            for idx in field.data:
                part = field.parts[idx]
                if inner_type == GGUFValueType.STRING:
                    items.append(str(bytes(part), encoding='utf-8'))
                else:
                    val = part[0]
                    if hasattr(val, 'item'): val = val.item()
                    items.append(val)
            writer.add_array(key, items)
        else:
            # Scalar values
            val = field.parts[field.data[0]][0]
            if hasattr(val, 'item'): val = val.item()
            writer.add_key_value(key, val, vtype)

    # 4) Copy tensors
    print("Processing tensors...")
    for tensor in reader.tensors:
        name = tensor.name
        qtype = tensor.tensor_type
        data = tensor.data
        
        if qtype == GGMLQuantizationType.I2_S:
            print(f"Transforming I2_S -> TL1: {name}")
            # 4) Read extra scale data and pass to custom function
            scale = tensor.scale
            K = int(tensor.shape[0])  # Width / Input features
            M = int(tensor.shape[1])  # Height / Output features
            new_data, new_scale = transform_i2_s_to_tl1(data, M, K, scale)
            
            # 5) Write as TL1 packed tensors
            writer.add_tensor(name, new_data, raw_shape=tensor.shape[::-1], raw_dtype=GGMLQuantizationType.TL1)
            print(f"Added TL1 tensor: {name} with shape {tensor.shape}")
            if new_scale is not None:
                # Add the hidden scale as an F32 tensor; GGUFWriter handles the embedding
                scale_arr = np.array([new_scale], dtype=np.float32)
                writer.add_tensor(name + "_scale", scale_arr, raw_dtype=GGMLQuantizationType.F32)
                print(f"   Added hidden scale tensor: {name}_scale with value {new_scale}")
        else:
            # 4) Copy tensors for all other types exactly as they are
            writer.add_tensor(name, data, raw_shape=tensor.shape[::-1], raw_dtype=qtype)
            print(f"Copied tensor: {name} with type {qtype} and shape {tensor.shape}")

    print(f"Finalizing and writing to: {output_path}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("Success!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert-i2_s-to-tl1.py <input_i2_s.gguf> <output_tl1.gguf>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    convert_model(input_file, output_file)
