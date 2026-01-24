import struct
from transformers import AutoTokenizer

model_path = "models/bitnet-b1.58-2B-4T-bf16/ggml-model-tl1.gguf"
model_dir = "models/bitnet-b1.58-2B-4T-bf16"

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    text = "Is capitalism good or bad?"

    # 1. Encode to IDs
    ids = tokenizer.encode(text, add_special_tokens=True)
    # 2. Decode back to string
    decoded_text = tokenizer.decode(ids)

    print(f"Original: {text}")
    print(f"IDs:      {ids}")
    print(f"Decoded:  {decoded_text}")

    return ids

def find_token_at_index(file_path, target_idx):
    with open(file_path, 'rb') as f:
        # Skip GGUF Header (Magic: 4, Version: 4, TensorCount: 8, KVCount: 8) = 24 bytes
        f.seek(24)
        
        # Search for the 'tokenizer.ggml.tokens' key in the first 50MB
        content = f.read(50 * 1024 * 1024)
        key = b"tokenizer.ggml.tokens"
        key_pos = content.find(key)
        
        if key_pos == -1:
            return "Key not found in the first 50MB."

        # The GGUF structure after the key is:
        # [4 bytes: GGUF_TYPE_ARRAY (9)] 
        # [4 bytes: GGUF_TYPE_STRING (8) - the type inside the array]
        # [8 bytes: Number of elements in array]
        # [Each element: 8 bytes length + N bytes string data]
        
        # Move to the start of the array data
        # Position is: key_pos + len(key) + type_info(8 bytes)
        pos = key_pos + len(key) + 8 
        n_elements = struct.unpack("<Q", content[pos:pos+8])[0]
        
        if target_idx >= n_elements:
            return f"Index {target_idx} out of range (Total: {n_elements})"
        
        curr_pos = pos + 8
        for i in range(n_elements):
            str_len = struct.unpack("<Q", content[curr_pos:curr_pos+8])[0]
            curr_pos += 8
            if i == target_idx:
                token_val = content[curr_pos:curr_pos+str_len].decode('utf-8', errors='replace')
                return f"Token[{i}]: {token_val}"
            curr_pos += str_len

#print(find_token_at_index(model_path, 1567))
ids = load_tokenizer()