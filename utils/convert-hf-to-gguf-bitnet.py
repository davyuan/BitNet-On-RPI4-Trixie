#!/usr/bin/env python3

from __future__ import annotations

import logging
import argparse
import contextlib
import json
import os
import re
import sys
from abc import ABC, abstractmethod
from enum import IntEnum
from pathlib import Path
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Iterator, Sequence, TypeVar, cast
import configparser

import numpy as np
import torch
from transformers import AutoTokenizer

if TYPE_CHECKING:   
    from torch import Tensor

if 'NO_LOCAL_GGUF' not in os.environ:
    gguf_path = Path(__file__).resolve().parent / '3rdparty' / 'llama.cpp' / 'gguf-py'
    if gguf_path.exists():
        sys.path.insert(1, str(gguf_path))
import gguf

from convert import LlamaHfVocab, permute

logger = logging.getLogger("hf-to-gguf")


###### MODEL DEFINITIONS ######

class SentencePieceTokenTypes(IntEnum):
    NORMAL = 1
    UNKNOWN = 2
    CONTROL = 3
    USER_DEFINED = 4
    UNUSED = 5
    BYTE = 6


AnyModel = TypeVar("AnyModel", bound="type[Model]")


class Model(ABC):
    _model_classes: dict[str, type[Model]] = {}

    def __init__(self, dir_model: Path, ftype: int, fname_out: Path, is_big_endian: bool, use_temp_file: bool):
        self.dir_model = dir_model
        self.ftype = ftype
        self.fname_out = fname_out
        self.is_big_endian = is_big_endian
        self.endianess = gguf.GGUFEndian.BIG if is_big_endian else gguf.GGUFEndian.LITTLE
        self.use_temp_file = use_temp_file
        self.is_safetensors = self._is_model_safetensors()
        self.num_parts = Model.count_model_parts(self.dir_model, ".safetensors" if self.is_safetensors else ".bin")
        self.part_names = self._get_part_names()
        self.hparams = Model.load_hparams(self.dir_model)
        self.gguf_writer = gguf.GGUFWriter(fname_out, gguf.MODEL_ARCH_NAMES[self.model_arch], endianess=self.endianess, use_temp_file=self.use_temp_file)
        self.block_count = self.find_hparam(["n_layers", "num_hidden_layers", "n_layer"])
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    @property
    @abstractmethod
    def model_arch(self) -> gguf.MODEL_ARCH:
        pass

    def find_hparam(self, keys: Sequence[str], optional: bool = False) -> Any:
        key = next((k for k in keys if k in self.hparams), None)
        if key is not None:
            return self.hparams[key]
        if optional:
            return None
        raise KeyError(f"could not find any of: {keys}")

    def set_vocab(self):
        self._set_vocab_gpt2()

    def get_tensors(self) -> Iterator[tuple[str, Tensor]]:
        for part_name in self.part_names:
            logger.info(f"gguf: loading model part '{part_name}'")
            ctx: ContextManager[Any]
            if self.is_safetensors:
                from safetensors import safe_open
                ctx = cast(ContextManager[Any], safe_open(self.dir_model / part_name, framework="pt", device="cpu"))
            else:
                ctx = contextlib.nullcontext(torch.load(str(self.dir_model / part_name), map_location="cpu", mmap=True, weights_only=True))

            with ctx as model_part:
                for name in model_part.keys():
                    data = model_part.get_tensor(name) if self.is_safetensors else model_part[name]
                    yield name, data

    def match_model_tensor_name(self, name: str, key: gguf.MODEL_TENSOR, bid: int | None, suffix: str = ".weight") -> bool:
        if key not in gguf.MODEL_TENSORS[self.model_arch]:
            return False
        key_name: str = gguf.TENSOR_NAMES[key]
        if "{bid}" in key_name:
            if bid is None:
                return False
            key_name = key_name.format(bid=bid)
        else:
            if bid is not None:
                return False
        return name == (key_name + suffix)

    def map_tensor_name(self, name: str, try_suffixes: Sequence[str] = (".weight", ".bias")) -> str:
        new_name = self.tensor_map.get_name(key=name, try_suffixes=try_suffixes)
        if new_name is None:
            raise ValueError(f"Can not map tensor {name!r}")
        return new_name

    def set_gguf_parameters(self):
        self.gguf_writer.add_name(self.dir_model.name)
        self.gguf_writer.add_block_count(self.block_count)

        if (n_ctx := self.find_hparam(["max_position_embeddings", "n_ctx"], optional=True)) is not None:
            self.gguf_writer.add_context_length(n_ctx)
            logger.info(f"gguf: context length = {n_ctx}")

        n_embd = self.find_hparam(["hidden_size", "n_embd"])
        self.gguf_writer.add_embedding_length(n_embd)
        logger.info(f"gguf: embedding length = {n_embd}")

        if (n_ff := self.find_hparam(["intermediate_size", "n_inner"], optional=True)) is not None:
            self.gguf_writer.add_feed_forward_length(n_ff)
            logger.info(f"gguf: feed forward length = {n_ff}")

        n_head = self.find_hparam(["num_attention_heads", "n_head"])
        self.gguf_writer.add_head_count(n_head)
        logger.info(f"gguf: head count = {n_head}")

        if (n_head_kv := self.hparams.get("num_key_value_heads")) is not None:
            self.gguf_writer.add_head_count_kv(n_head_kv)
            logger.info(f"gguf: key-value head count = {n_head_kv}")

        if (rope_theta := self.hparams.get("rope_theta")) is not None:
            self.gguf_writer.add_rope_freq_base(rope_theta)
            logger.info(f"gguf: rope theta = {rope_theta}")
        if (f_rms_eps := self.hparams.get("rms_norm_eps")) is not None:
            self.gguf_writer.add_layer_norm_rms_eps(f_rms_eps)
            logger.info(f"gguf: rms norm epsilon = {f_rms_eps}")
        if (f_norm_eps := self.find_hparam(["layer_norm_eps", "layer_norm_epsilon", "norm_epsilon"], optional=True)) is not None:
            self.gguf_writer.add_layer_norm_eps(f_norm_eps)
            logger.info(f"gguf: layer norm epsilon = {f_norm_eps}")
        if (n_experts := self.hparams.get("num_local_experts")) is not None:
            self.gguf_writer.add_expert_count(n_experts)
            logger.info(f"gguf: expert count = {n_experts}")
        if (n_experts_used := self.hparams.get("num_experts_per_tok")) is not None:
            self.gguf_writer.add_expert_used_count(n_experts_used)
            logger.info(f"gguf: experts used count = {n_experts_used}")

        self.gguf_writer.add_file_type(self.ftype)
        logger.info(f"gguf: file type = {self.ftype}")

        self.gguf_writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
        logger.info(f"gguf: quantization version = {gguf.GGML_QUANT_VERSION}")

    def write_tensors(self):
        block_count = self.hparams.get("n_layers", self.hparams.get("num_hidden_layers", self.hparams.get("n_layer")))
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)
        for name, data_torch in self.get_tensors():
            # we don't need these
            if name.endswith((".attention.masked_bias", ".attention.bias", ".attention.rotary_emb.inv_freq")):
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            data = data_torch.squeeze().numpy()

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                raise ValueError(f"Can not map tensor {name!r}")

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and (n_dims == 1 or new_name.endswith("_norm.weight")):
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            logger.info(f"{new_name}, n_dims = {n_dims}, {old_dtype} --> {data.dtype}")

            self.gguf_writer.add_tensor(new_name, data)

    def write(self):
        self.write_tensors()
        self.gguf_writer.write_header_to_file()
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.write_tensors_to_file()
        self.gguf_writer.close()

    def write_vocab(self):
        self.gguf_writer.write_header_to_file()
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.close()

    @staticmethod
    def count_model_parts(dir_model: Path, prefix: str) -> int:
        num_parts = 0
        for filename in os.listdir(dir_model):
            if filename.endswith(prefix):
                num_parts += 1

        return num_parts

    @staticmethod
    def load_hparams(dir_model):
        with open(dir_model / "config.json", "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def register(cls, *names: str) -> Callable[[AnyModel], AnyModel]:
        assert names

        def func(modelcls: type[Model]):
            for name in names:
                cls._model_classes[name] = modelcls
            return modelcls
        return func

    @classmethod
    def from_model_architecture(cls, arch):
        try:
            return cls._model_classes[arch]
        except KeyError:
            raise NotImplementedError(f'Architecture {arch!r} not supported!') from None

    def _is_model_safetensors(self) -> bool:
        return Model.count_model_parts(self.dir_model, ".safetensors") > 0

    def _get_part_names(self):
        if self.is_safetensors:
            if self.num_parts == 1:  # there's only one .safetensors file
                return ("model.safetensors",)
            return (f"model-{n:05}-of-{self.num_parts:05}.safetensors" for n in range(1, self.num_parts + 1))

        if self.num_parts == 1:  # there's only one .bin file
            return ("pytorch_model.bin",)
        return (f"pytorch_model-{n:05}-of-{self.num_parts:05}.bin" for n in range(1, self.num_parts + 1))

    # used for GPT-2 BPE and WordPiece vocabs
    def get_vocab_base(self) -> tuple[list[str], list[int], str]:
        tokens: list[str] = []
        toktypes: list[int] = []

        # Load tokenizer from the BitNet model folder
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
        vocab_size = self.hparams.get("vocab_size", len(tokenizer.vocab))
        assert max(tokenizer.vocab.values()) < vocab_size

        # BitNet uses Llama-compatible BPE tokenization
        tokpre = "llama-bpe"

        reverse_vocab = {id_: encoded_tok for encoded_tok, id_ in tokenizer.vocab.items()}
        added_vocab = tokenizer.get_added_vocab()

        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.USER_DEFINED)
            elif reverse_vocab[i] in added_vocab:
                # For special tokens, use the raw token text directly
                # Don't encode/decode as it can cause merging of token sequences
                tokens.append(reverse_vocab[i])
                if tokenizer.added_tokens_decoder[i].special:
                    toktypes.append(gguf.TokenType.CONTROL)
                else:
                    toktypes.append(gguf.TokenType.USER_DEFINED)
            else:
                tokens.append(reverse_vocab[i])
                toktypes.append(gguf.TokenType.NORMAL)
        
        return tokens, toktypes, tokpre

    def _set_vocab_gpt2(self) -> None:
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_sentencepiece(self):
        tokenizer_model_path = self.dir_model / 'tokenizer.model'
        tokenizer_json_path = self.dir_model / 'tokenizer.json'

        tokens: list[bytes] = []
        scores: list[float] = []
        toktypes: list[int] = []

        # Check which tokenizer file exists
        if tokenizer_json_path.is_file():
            # Load from tokenizer.json using transformers
            from transformers import AutoTokenizer
            
            logger.info(f"Loading tokenizer from {tokenizer_json_path}")
            tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
            vocab_size = self.hparams.get('vocab_size', len(tokenizer.vocab))
            
            # Read tokenizer.json for scores
            with open(tokenizer_json_path, "r", encoding="utf-8") as f:
                tokenizer_json = json.load(f)
            
            # Extract vocab and scores from tokenizer.json
            vocab_scores = {}
            if "model" in tokenizer_json and "vocab" in tokenizer_json["model"]:
                # For SentencePiece models, vocab is a list of [token, score] pairs
                vocab_list = tokenizer_json["model"]["vocab"]
                if isinstance(vocab_list, list):
                    for item in vocab_list:
                        if isinstance(item, list) and len(item) >= 2:
                            token, score = item[0], item[1]
                            vocab_scores[token] = score
                elif isinstance(vocab_list, dict):
                    vocab_scores = vocab_list
            
            # Build the tokens list
            reverse_vocab = {id_: token for token, id_ in tokenizer.vocab.items()}
            added_vocab = tokenizer.get_added_vocab()
            
            for i in range(vocab_size):
                if i not in reverse_vocab:
                    tokens.append(f"[PAD{i}]".encode("utf-8"))
                    scores.append(-1000.0)
                    toktypes.append(SentencePieceTokenTypes.UNUSED)
                else:
                    token = reverse_vocab[i]
                    # Ensure token is a string for vocab_scores lookup
                    if isinstance(token, bytes):
                        token_str = token.decode("utf-8")
                        tokens.append(token)
                    else:
                        token_str = token
                        tokens.append(token.encode("utf-8"))
                    # Get score from tokenizer.json or use default
                    scores.append(float(vocab_scores.get(token_str, 0.0)))
                    
                    # Determine token type
                    if token_str in added_vocab:
                        if tokenizer.added_tokens_decoder[i].special:
                            toktypes.append(SentencePieceTokenTypes.CONTROL)
                        else:
                            toktypes.append(SentencePieceTokenTypes.USER_DEFINED)
                    else:
                        toktypes.append(SentencePieceTokenTypes.NORMAL)
        
        elif tokenizer_model_path.is_file():
            # Fallback to original SentencePiece loading
            from sentencepiece import SentencePieceProcessor
            
            logger.info(f"Loading tokenizer from {tokenizer_model_path}")
            tokenizer = SentencePieceProcessor(str(tokenizer_model_path))
            vocab_size = self.hparams.get('vocab_size', tokenizer.vocab_size())

            for token_id in range(tokenizer.vocab_size()):
                piece = tokenizer.id_to_piece(token_id)
                text = piece.encode("utf-8")
                score = tokenizer.get_score(token_id)

                toktype = SentencePieceTokenTypes.NORMAL
                if tokenizer.is_unknown(token_id):
                    toktype = SentencePieceTokenTypes.UNKNOWN
                elif tokenizer.is_control(token_id):
                    toktype = SentencePieceTokenTypes.CONTROL
                elif tokenizer.is_unused(token_id):
                    toktype = SentencePieceTokenTypes.UNUSED
                elif tokenizer.is_byte(token_id):
                    toktype = SentencePieceTokenTypes.BYTE

                tokens.append(text)
                scores.append(score)
                toktypes.append(toktype)
        else:
            raise FileNotFoundError(f"No tokenizer file found: {tokenizer_json_path} or {tokenizer_model_path}")

        added_tokens_file = self.dir_model / 'added_tokens.json'
        if added_tokens_file.is_file():
            with open(added_tokens_file, "r", encoding="utf-8") as f:
                added_tokens_json = json.load(f)

                for key in added_tokens_json:
                    key = key.encode("utf-8")
                    if key not in tokens:
                        tokens.append(key)
                        scores.append(-1000.0)
                        toktypes.append(SentencePieceTokenTypes.USER_DEFINED)

        if vocab_size > len(tokens):
            pad_count = vocab_size - len(tokens)
            logger.debug(f"Padding vocab with {pad_count} token(s) - [PAD1] through [PAD{pad_count}]")
            for i in range(1, pad_count + 1):
                tokens.append(f"[PAD{i}]".encode("utf-8"))
                scores.append(-1000.0)
                toktypes.append(SentencePieceTokenTypes.UNUSED)

        assert len(tokens) == vocab_size

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_llama_hf(self):
        vocab = LlamaHfVocab(self.dir_model)
        tokens = []
        scores = []
        toktypes = []

        for text, score, toktype in vocab.all_tokens():
            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)

        assert len(tokens) == vocab.vocab_size

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

# TL1

def process_tl1(weight, M, K):
    weight = weight.transpose(1, 0)  # K/2, M
    weight = weight.reshape((K // 2, M // 2, 2))
    weight_0 = weight[:, :, 0] << 4
    weight_1 = weight[:, :, 1]
    weight = weight_0 + weight_1
    return weight  #Shape (K/2, M/2)

def preprocess_weights_tl1(
    w: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:

    M, K = w.shape
    weight, scale = bitnet_158_quantize(w)
    weight_num = M * K
    weight = np.reshape(weight, (weight_num // 2, 2))
    hi_weight = np.multiply(np.split(weight, 2, axis=1)[0], 3)
    lo_weight = np.split(weight, 2, axis=1)[1]

    weight = np.reshape((hi_weight + lo_weight), weight_num // 2)

    weight = weight + 4
    weight = np.reshape(weight, (M, K // 2))
    '''print("First 32 rows of weight before packing (32 elements each, hex):")
    for i in range(min(32, weight.shape[0])):
        row_hex = ' '.join(f'0x{x:02x}' for x in weight[i, :32])
        print(row_hex)'''

    weight = process_tl1(weight, M, K)
    '''print("First 16 rows of weight after packing (16 elements each, hex):")
    for i in range(min(16, weight.shape[0])):
        row_hex = ' '.join(f'0x{x:02x}' for x in weight[i, :16])
        print(row_hex)
    sj = input("Press Enter to continue...")'''

    return weight, scale  #Shape (K/2, M/2), ()

def bitnet_158_quantize(weight_array):
    """
    Quantizes a weight matrix to ternary values {-1, 0, 1} 
    using the absmean scaling and RoundClip method.
    """
    gamma = np.mean(np.abs(weight_array))
    epsilon = 1e-7
    normalized_w = weight_array / (gamma + epsilon)
    quantized_w = np.clip(np.round(normalized_w), -1, 1).astype(np.int8)
    
    return quantized_w, gamma

def preprocess_two_weights_tl2(M, K, weight_num, BM, BY, bm, by, weight, final_weight):
    weight = np.reshape(weight, (weight_num // 2, 2))
    hi_weight = np.multiply(np.split(weight, 2, axis=1)[0], 3)
    lo_weight = np.split(weight, 2, axis=1)[1]

    weight = np.reshape((hi_weight + lo_weight), weight_num // 2)

    weight = weight + 4
    weight = np.reshape(weight, (M, K // 2)).astype(np.uint8)
    weight = weight.reshape((M // BM, BM, K // 2)).transpose(0, 2, 1)
    weight = weight.reshape((M // BM, K // BY, BY // 2, BM)).transpose(0, 1, 3, 2)
    weight = weight.reshape((M // BM, K // BY, BM // bm, bm, BY // 2)).transpose(0, 1, 2, 4, 3)
    weight = weight.reshape((M // BM, K // BY, BM // bm, BY // by, by // 2, bm)).transpose(0, 1, 2, 3, 5, 4)
    weight = weight.reshape((M // BM, K // BY, BM // bm, BY // by, bm, by // 2))
    weight_0 = weight[:, :, :, :, :, 0]
    weight_1 = weight[:, :, :, :, :, 1]
    weight_0 = weight_0 << 4
    weight_1 = weight_1
    weight = weight_0 + weight_1
    weight = weight.reshape((M * K // bm // by, bm // 8, 8))
    weight[:, [0, 1, 2, 3], :] = weight[:, [0, 2, 1, 3], :]
    weight = weight.reshape(M * K // bm // by, bm)
    
    for i in range(weight.shape[0]):
        final_weight.append(weight[i, :])

def preprocess_three_weights_tl2(M, K, weight_num, BM, BY, bm, by, weight, final_weight):
    weight = np.reshape(weight, (weight_num // 3, 3))
    split_weights = np.split(weight, 3, axis=1)
    first_weight = np.multiply(split_weights[0], 9)
    second_weight = np.multiply(split_weights[1], 3)
    third_weight = split_weights[2]

    weight = np.reshape((first_weight + second_weight + third_weight), weight_num // 3)
    sign_weight = np.sign(weight) + 2
    sign_weight = np.where(sign_weight > 1, 0, sign_weight)
    weight = np.abs(weight)

    weight = np.reshape(weight, (M, K // 3)).astype(np.uint8)
    sign_weight = np.reshape(sign_weight, (M, K // 3)).astype(np.uint8)

    weight = weight.reshape((M // BM, BM, K // 3)).transpose(0, 2, 1)
    weight = weight.reshape((M // BM, K // BY, BY // 3, BM)).transpose(0, 1, 3, 2)
    weight = weight.reshape((M // BM, K // BY, BM // bm, bm, BY // 3)).transpose(0, 1, 2, 4, 3)
    weight = weight.reshape((M // BM, K // BY, BM // bm, BY // by, by // 3, bm)).transpose(0, 1, 2, 3, 5, 4)
    weight = weight.reshape((M // BM, K // BY, BM // bm, BY // by, bm, by // 3))
    weight_0 = weight[:, :, :, :, :, 0]
    weight_1 = weight[:, :, :, :, :, 1]
    weight_0 = weight_0 << 4
    weight_1 = weight_1
    weight = weight_0 + weight_1
    weight = weight.reshape((M * K // bm // by, bm // 8, 8))
    weight[:, [0, 1, 2, 3], :] = weight[:, [0, 2, 1, 3], :]
    weight = weight.reshape(M * K // bm // by, bm)

    for i in range(weight.shape[0]):
        final_weight.append(weight[i, :])

    sign_weight = sign_weight.reshape((M // BM, BM, K // 3)).transpose(0, 2, 1)
    sign_weight = sign_weight.reshape((M // BM, K // BY, BY // 3, BM)).transpose(0, 1, 3, 2)
    sign_weight = sign_weight.reshape((M // BM, K // BY, BM // bm, bm, BY // 3)).transpose(0, 1, 2, 4, 3)
    sign_weight = sign_weight.reshape((M // BM, K // BY, BM // bm, BY // (by * 4), by // 3 * 4, bm)).transpose(0, 1, 2, 3, 5, 4)
    sign_weight = sign_weight.reshape((M // BM, K // BY, BM // bm, BY // (by * 4), bm, by // 3 * 4)).transpose(0, 1, 2, 3, 5, 4)
    sign_weight = sign_weight.reshape((M // BM, K // BY, BM // bm, BY // (by * 4), by // 3 * 8, bm // 2)).astype(np.uint16)
    combine_weight = np.zeros((M // BM, K // BY, BM // bm, BY // (by * 4), bm // 2), dtype=np.uint16)
    for i in range(16):
        temp_weight = sign_weight[:, :, :, :, i, :] << 15 - i
        combine_weight += temp_weight
    combine_weight = combine_weight.view(np.uint8)
    combine_weight = combine_weight.reshape((M * K // bm // (by * 4)), bm)
    
    for i in range(combine_weight.shape[0]):
        final_weight.append(combine_weight[i, :])

def preprocess_weights_tl2(
    w: np.ndarray,
    bits = 2,
    g    = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    from configparser import ConfigParser
    config = ConfigParser()

    M, K = w.shape
    weight = w
    weight = np.where(np.abs(weight) < 1e-6, 0, weight).astype(np.float32)
    weight = np.sign(weight)
    weight_num = np.prod(weight.shape)

    config.read('include/kernel_config.ini')
    BM = -1
    BY = -1
    bm = -1

    for kernel in config.sections():
        if int(config.get(kernel, 'm')) == M and int(config.get(kernel, 'k')) == K:
            BM = int(config.get(kernel, 'bm'))
            BY = int(config.get(kernel, 'bk'))
            bm = int(config.get(kernel, 'bmm'))
            by = 192 // bm
            break

    if BM == -1:
        raise NotImplementedError

    if (weight.shape[1] % BY != 0):
        slice_k_idx = weight.shape[1] - weight.shape[1] % BY
        slice_weights = np.split(weight, [slice_k_idx], axis=1)
        three_weight = slice_weights[0]
        two_weight = slice_weights[1]
    else:
        three_weight = weight

    final_weight = []

    preprocess_three_weights_tl2(three_weight.shape[0],
                         three_weight.shape[1],
                         three_weight.shape[0] * three_weight.shape[1],
                         BM,
                         BY,
                         bm,
                         by,
                         three_weight,
                         final_weight)

    if (weight.shape[1] % BY != 0):
        preprocess_two_weights_tl2(  two_weight.shape[0],
                         two_weight.shape[1],
                         two_weight.shape[0] * two_weight.shape[1],
                         BM,
                         32,
                         32,
                         4,
                         two_weight,
                         final_weight)
    weight = np.array(final_weight, dtype=np.uint8).reshape(-1)
    weight = np.pad(weight, (0, (K - 256) * M // 3 * 5 // 8 + 256 * M // 2 * 4 // 8 -
                             weight.shape[0]), mode='constant', constant_values=0)
    return weight

def transform_to_tl1(x: np.ndarray):
    res, scale = preprocess_weights_tl1(x)
    return res, np.asarray(scale, dtype=np.float32)

def transform_to_tl2(x: np.ndarray):
    scale = np.max(np.abs(x))
    # res = np.round(x / scale + 2).astype(np.uint8)
    res = preprocess_weights_tl2(x)
    return res, scale


def read_model_config(model_dir: str) -> dict[str, Any]:
    config = os.path.join(model_dir, "config.json")
    if not os.path.exists(config):
        raise FileNotFoundError(f"Model config file not found: {config}")
    with open(config, "r") as f:
        return json.load(f)

@Model.register("LLaMAForCausalLM", "LlamaForCausalLM", "MistralForCausalLM", "MixtralForCausalLM")
class LlamaModel(Model):
    model_arch = gguf.MODEL_ARCH.LLAMA

    def set_vocab(self):
        try:
            self._set_vocab_sentencepiece()
        except FileNotFoundError:
            try:
                self._set_vocab_llama_hf()
            except (FileNotFoundError, TypeError):
                # Llama 3
                self._set_vocab_gpt2()

        # Apply to CodeLlama only (and ignore for Llama 3 with a vocab size of 128256)
        if self.hparams.get("vocab_size", 32000) == 32016:
            special_vocab = gguf.SpecialVocab(
                self.dir_model, load_merges=False,
                special_token_types = ['prefix', 'suffix', 'middle', 'eot']
            )
            special_vocab._set_special_token("prefix", 32007)
            special_vocab._set_special_token("suffix", 32008)
            special_vocab._set_special_token("middle", 32009)
            special_vocab._set_special_token("eot",    32010)
            special_vocab.add_to_gguf(self.gguf_writer)

    def write_tensors(self):
        max_name_len = max(len(s) for _, s in self.tensor_map.mapping.values()) + len(".weight,")

        scale_map = dict()

        for name, data_torch in self.get_tensors():
            if name.endswith(("weight_scale")):
                data_torch = data_torch.to(torch.float32)
                name = name.replace(".weight_scale", "")
                scale_map[name] = data_torch

        for name, data_torch in self.get_tensors():
            if name.endswith(("weight_scale")):
                continue
            # we don't need these
            if name.endswith((".attention.masked_bias", ".attention.bias", ".rotary_emb.inv_freq")):
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            if name.replace(".weight", "") in scale_map:
                data_torch = data_torch.to(torch.uint8)
                origin_shape = data_torch.shape
                shift = torch.tensor([0, 2, 4, 6], dtype=torch.uint8).reshape((4, *(1 for _ in range(len(origin_shape)))))
                data_torch = data_torch.unsqueeze(0).expand((4, *origin_shape)) >> shift
                data_torch = data_torch & 3
                data_torch = (data_torch.float() - 1).reshape((origin_shape[0] * 4, *origin_shape[1:]))
                data_torch = data_torch / scale_map[name.replace(".weight", "")].float()

            # use the first number-like part of the tensor name as the block id
            bid = None
            for part in name.split("."):
                if part.isdecimal():
                    bid = int(part)
                    break

            # old gguf bf16 not implenmented
            # if data_torch.dtype == torch.bfloat16:
            #     for new_name, data in ((n, d) for n, d in self.modify_tensors(data_torch, name, bid)):
            #         shape_str = f"{{{', '.join(str(n) for n in reversed(data.shape))}}}"
            #         # n_dims is implicit in the shape
            #         logger.info(f"{f'%-{max_name_len}s' % f'{new_name},'} {old_dtype}, shape = {shape_str}")
            #         self.gguf_writer.add_tensor(new_name, data, raw_shape=data.shape, raw_dtype=data.dtype)
            #     continue

            for new_name, data in ((n, d.squeeze().numpy()) for n, d in self.modify_tensors(data_torch, name, bid)):
                data: np.ndarray = data  # type hint
                data_shape = data.shape
                shape_before_quant = data_shape
                n_dims = len(data.shape)
                data_dtype = data.dtype
                data_qtype: gguf.GGMLQuantizationType | None = None

                # when both are True, f32 should win
                # extra_f32 = self.extra_f32_tensors(name, new_name, bid, n_dims)
                # extra_f16 = self.extra_f16_tensors(name, new_name, bid, n_dims)
                extra_f32 = False
                extra_f16 = False

                # Most of the codebase that takes in 1D tensors or norms only handles F32 tensors
                # Conditions should closely match those in llama_model_quantize_internal in llama.cpp
                extra_f32 = any(cond for cond in (
                    extra_f32,
                    n_dims == 1,
                    new_name.endswith("_norm.weight"),
                ))

                # Some tensor types are always in float32
                tensors_f32 = [
                    gguf.MODEL_TENSOR.FFN_GATE_INP,
                    gguf.MODEL_TENSOR.FFN_GATE_INP,
                    gguf.MODEL_TENSOR.POS_EMBD,
                    gguf.MODEL_TENSOR.TOKEN_TYPES,
                ]
                extra_f32 = extra_f32 or any(self.match_model_tensor_name(new_name, key, bid) for key in tensors_f32)

                # if f16 desired, convert any float32 2-dim weight tensors to float16
                extra_f16 = any(cond for cond in (
                    extra_f16,
                    (name.endswith(".weight") and n_dims >= 2),
                ))

                suit_i2 = True
                if name.endswith('lm_head.weight') or name.endswith('norm.weight') or name.endswith('embed_tokens.weight'):
                    suit_i2 = False

                i2_scale = None
                if self.ftype != gguf.GGMLQuantizationType.F32 and extra_f16 and not extra_f32:
                    if self.ftype == gguf.GGMLQuantizationType.TL1 and suit_i2:
                        data, i2_scale = transform_to_tl1(data)
                        assert data.dtype == np.uint8
                        assert i2_scale.dtype == np.float32
                        data_qtype = gguf.GGMLQuantizationType.TL1
                    elif self.ftype == gguf.GGMLQuantizationType.TL2 and suit_i2:
                        data, i2_scale = transform_to_tl2(data)
                        assert data.dtype == np.uint8
                        assert i2_scale.dtype == np.float32
                        data_qtype = gguf.GGMLQuantizationType.TL2
                    else:  # default to float16 for quantized tensors
                        if data_dtype != np.float16:
                            data = data.astype(np.float16)
                        data_qtype = gguf.GGMLQuantizationType.F16

                if data_qtype is None:  # by default, convert to float32
                    if data_dtype != np.float32:
                        data = data.astype(np.float32)
                    data_qtype = gguf.GGMLQuantizationType.F32

                shape = data.shape
                # reverse shape to make it similar to the internal ggml dimension order
                shape_str = f"{{{', '.join(str(n) for n in reversed(shape))}}}"

                # n_dims is implicit in the shape
                logger.info(f"{f'%-{max_name_len}s' % f'{new_name},'} {old_dtype} --> {data_qtype.name}, shape = {shape_str}")
                
                # For TL1/TL2, pass the original logical shape so llama.cpp gets the right tensor dimensions
                raw_shape = shape_before_quant if data_qtype in (gguf.GGMLQuantizationType.TL1, gguf.GGMLQuantizationType.TL2) else data.shape
                self.gguf_writer.add_tensor(new_name, data, raw_shape=raw_shape, raw_dtype=data_qtype)
                if i2_scale is not None:
                    self.gguf_writer.add_tensor(new_name + "_scale", i2_scale, raw_dtype=gguf.GGMLQuantizationType.F32)
                    logger.info(f"    Added scale tensor: {new_name + '_scale'}, F32, value: {i2_scale.item()}")


    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])

        if "head_dim" in hparams:
            rope_dim = hparams["head_dim"]
        else:
            rope_dim = hparams["hidden_size"] // hparams["num_attention_heads"]
        self.gguf_writer.add_rope_dimension_count(rope_dim)

        if self.hparams.get("rope_scaling") is not None and "factor" in self.hparams["rope_scaling"]:
            if self.hparams["rope_scaling"].get("type") == "linear":
                self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
                self.gguf_writer.add_rope_scaling_factor(self.hparams["rope_scaling"]["factor"])

        tokenizer_config_file = self.dir_model / 'tokenizer_config.json'
        if tokenizer_config_file.is_file():
            with open(tokenizer_config_file, "r", encoding="utf-8") as f:
                tokenizer_config_json = json.load(f)
                if "add_prefix_space" in tokenizer_config_json:
                    self.gguf_writer.add_add_space_prefix(tokenizer_config_json["add_prefix_space"])

        # Apply to granite small models only
        if self.hparams.get("vocab_size", 32000) == 49152:
            self.gguf_writer.add_add_bos_token(False)

    @staticmethod
    def permute(weights: Tensor, n_head: int, n_head_kv: int | None):
        if n_head_kv is not None and n_head != n_head_kv:
            n_head = n_head_kv
        return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                .swapaxes(1, 2)
                .reshape(weights.shape))

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        n_head = self.hparams["num_attention_heads"]
        n_kv_head = self.hparams.get("num_key_value_heads")

        if name.endswith(("q_proj.weight", "q_proj.bias")):
            data_torch = LlamaModel.permute(data_torch, n_head, n_head)
        if name.endswith(("k_proj.weight", "k_proj.bias")):
            data_torch = LlamaModel.permute(data_torch, n_head, n_kv_head)

        # process the experts separately
        if name.find("block_sparse_moe.experts") != -1:
            n_experts = self.hparams["num_local_experts"]

            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                tensors: list[tuple[str, Tensor]] = []

                # merge the experts into a single 3d tensor
                for wid in ["w1", "w2", "w3"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.block_sparse_moe.experts.{xid}.{wid}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"layers.{bid}.feed_forward.experts.{wid}.weight"

                    new_name = self.map_tensor_name(merged_name)

                    tensors.append((new_name, data_torch))
                return tensors
            else:
                return []

        return [(self.map_tensor_name(name), data_torch)]

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        if rope_scaling := self.find_hparam(["rope_scaling"], optional=True):
            if rope_scaling.get("rope_type", '').lower() == "llama3":
                base = self.hparams.get("rope_theta", 10000.0)
                dim = self.hparams.get("head_dim", self.hparams["hidden_size"] // self.hparams["num_attention_heads"])
                freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

                factor = rope_scaling.get("factor", 8.0)
                low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
                high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
                old_context_len = self.hparams.get("original_max_position_embeddings", 8192)

                low_freq_wavelen = old_context_len / low_freq_factor
                high_freq_wavelen = old_context_len / high_freq_factor
                assert low_freq_wavelen != high_freq_wavelen

                rope_factors = []
                for freq in freqs:
                    wavelen = 2 * math.pi / freq
                    if wavelen < high_freq_wavelen:
                        rope_factors.append(1)
                    elif wavelen > low_freq_wavelen:
                        rope_factors.append(factor)
                    else:
                        smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                        rope_factors.append(1 / ((1 - smooth) / factor + smooth))

                yield (self.format_tensor_name(gguf.MODEL_TENSOR.ROPE_FREQS), torch.tensor(rope_factors, dtype=torch.float32))

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")


@Model.register("BitnetForCausalLM", "BitNetForCausalLM")
class BitnetModel(Model):
    model_arch = gguf.MODEL_ARCH.BITNET_B158

    def get_vocab_base(self) -> tuple[list[str], list[int], str]:
        """Override to fix Mistral/Llama tokenizer regex issues"""
        tokens: list[str] = []
        toktypes: list[int] = []

        from transformers import AutoTokenizer
        # Use fix_mistral_regex=True to handle non-standard tokenizers
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model, fix_mistral_regex=True)
        vocab_size = self.hparams.get("vocab_size", len(tokenizer.vocab))
        assert max(tokenizer.vocab.values()) < vocab_size

        tokpre = self.get_vocab_base_pre(tokenizer)

        reverse_vocab = {id_: encoded_tok for encoded_tok, id_ in tokenizer.vocab.items()}
        added_vocab = tokenizer.get_added_vocab()

        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.USER_DEFINED)
            elif reverse_vocab[i] in added_vocab:
                # For special tokens, use the raw token text directly
                # Don't encode/decode as it can cause merging of token sequences
                tokens.append(reverse_vocab[i])
                if tokenizer.added_tokens_decoder[i].special:
                    toktypes.append(gguf.TokenType.CONTROL)
                else:
                    toktypes.append(gguf.TokenType.USER_DEFINED)
            else:
                tokens.append(reverse_vocab[i])
                toktypes.append(gguf.TokenType.NORMAL)
        return tokens, toktypes, tokpre

    def get_vocab_base_pre(self, tokenizer) -> str:
        """Override to handle non-standard tokenizers gracefully"""
        try:
            # Try the parent implementation first
            return super().get_vocab_base_pre(tokenizer)
        except (NotImplementedError, AttributeError):
            # If the hash doesn't match or method doesn't exist, detect based on tokenizer config
            logger.warning("BPE pre-tokenizer hash not recognized, attempting to auto-detect...")
            
            # Check tokenizer config for hints
            tokenizer_config_path = self.dir_model / 'tokenizer_config.json'
            if tokenizer_config_path.is_file():
                try:
                    with open(tokenizer_config_path, "r", encoding="utf-8") as f:
                        tokenizer_config = json.load(f)
                    
                    # Detect based on model type in config
                    if any(key in str(tokenizer_config).lower() for key in ["llama", "meta-llama"]):
                        logger.info("Detected Llama-based tokenizer, using llama-bpe")
                        return "llama-bpe"
                    elif any(key in str(tokenizer_config).lower() for key in ["mistral"]):
                        logger.info("Detected Mistral-based tokenizer, using default BPE")
                        return "gpt-2"
                except (json.JSONDecodeError, IOError):
                    pass
            
            # Default fallback
            logger.warning("Defaulting to llama-bpe for BitNet model")
            return "llama-bpe"

    def _set_vocab_llama_hf(self):
        vocab = LlamaHfVocab(self.dir_model)
        tokens = []
        scores = []
        toktypes = []

        for text, score, toktype in vocab.all_tokens():
            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)

        assert len(tokens) == vocab.vocab_size

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_vocab(self):
        # Auto-detect tokenizer type from config and tokenizer.json structure
        tokenizer_json_path = self.dir_model / 'tokenizer.json'
        tokenizer_model_path = self.dir_model / 'tokenizer.model'
        vocab_json_path = self.dir_model / 'vocab.json'
        merges_txt_path = self.dir_model / 'merges.txt'
        tokenizer_config_path = self.dir_model / 'tokenizer_config.json'
        config_path = self.dir_model / 'config.json'
        
        is_bpe = False
        is_llama_hf = False
        
        # FIRST: Check if this is a Llama-style HF tokenizer (check tokenizer_config.json)
        # This takes priority over BPE detection since Llama HF can also have merges
        # Note: Exclude BitNet here as it needs BpeVocab, not LlamaHfVocab
        if tokenizer_config_path.is_file():
            try:
                with open(tokenizer_config_path, "r", encoding="utf-8") as f:
                    tokenizer_config = json.load(f)
                # If it's Llama/Mistral-based (but not BitNet), use the HF vocab handler
                if any(key in str(tokenizer_config).lower() for key in ["llama", "mistral", "meta-llama"]):
                    logger.info("Detected Llama/Mistral HF tokenizer - using HF vocab handler")
                    is_llama_hf = True
            except (json.JSONDecodeError, IOError):
                pass
        
        # Only check for BPE if NOT already detected as Llama HF
        if not is_llama_hf:
            # First check for explicit BPE files (legacy setup)
            if vocab_json_path.is_file() and merges_txt_path.is_file():
                logger.info("Detected BPE tokenizer (vocab.json + merges.txt)")
                is_bpe = True
            
            # Check tokenizer.json structure to detect BPE or SentencePiece
            elif tokenizer_json_path.is_file():
                try:
                    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
                        tokenizer_json = json.load(f)
                    
                    # BPE tokenizers have "model" field with "type": "BPE" or similar
                    if "model" in tokenizer_json:
                        model_config = tokenizer_json["model"]
                        if isinstance(model_config, dict) and model_config.get("type") in ("BPE", "WordPiece"):
                            logger.info(f"Detected {model_config.get('type')} tokenizer from tokenizer.json")
                            is_bpe = True
                    
                    # Additional heuristic: BPE models have merges, SentencePiece doesn't
                    if not is_bpe and "model" in tokenizer_json:
                        model_config = tokenizer_json["model"]
                        if isinstance(model_config, dict) and "merges" in model_config:
                            logger.info("Detected BPE tokenizer (merges field found in tokenizer.json)")
                            is_bpe = True
                            
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Could not parse tokenizer.json: {e}")
        
        # Use detected or default tokenizer type
        if is_llama_hf:
            logger.info("Using Llama/Mistral HF tokenizer (GGUF type: llama)")
            self._set_vocab_llama_hf()
        elif is_bpe:
            logger.info("Using BPE tokenizer (GGUF type: gpt2)")
            self._set_vocab_gpt2()
        elif tokenizer_model_path.is_file() or tokenizer_json_path.is_file():
            logger.info("Using SentencePiece tokenizer (GGUF type: llama)")
            self._set_vocab_sentencepiece()
        else:
            logger.warning("No tokenizer files detected, defaulting to SentencePiece")
            self._set_vocab_sentencepiece()

    def _set_vocab_gpt2(self) -> None:
        """Override GPT2 vocab to correctly set BitNet special tokens"""
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        # For BitNet, load and add merges
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        
        # Add merges if present
        if special_vocab.merges:
            self.gguf_writer.add_array("tokenizer.ggml.merges", special_vocab.merges)
            logger.info(f"gguf: added {len(special_vocab.merges)} merges")
        
        # Add chat template
        custom_template = (
            "{{ bos_token }}"
            "{% for message in messages %}"
            "{% set role = message['role'] | capitalize %}"
            "{% set content = message['content'] | trim %}"
            "{{ role + ': ' + content + '<|eot_id|>\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ 'Assistant: ' }}"
            "{% endif %}"
        )
        self.gguf_writer.add_string("tokenizer.chat_template", custom_template)
        
        # Add the special token IDs
        self.gguf_writer.add_uint32("tokenizer.ggml.bos_token_id", 128000)
        self.gguf_writer.add_uint32("tokenizer.ggml.eos_token_id", 128001)
        self.gguf_writer.add_uint32("tokenizer.ggml.eot_token_id", 128009)
        self.gguf_writer.add_uint32("tokenizer.ggml.padding_token_id", 128001)
        self.gguf_writer.add_bool("tokenizer.ggml.add_bos_token", True)
        
        
        
    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        self.gguf_writer.add_vocab_size(self.hparams["vocab_size"])

        # Add rope dimension count
        if "head_dim" in self.hparams:
            rope_dim = self.hparams["head_dim"]
        else:
            rope_dim = self.hparams["hidden_size"] // self.hparams["num_attention_heads"]
        self.gguf_writer.add_rope_dimension_count(rope_dim)

        # Extract model size from directory name or calculate from parameters
        # Look for size indicators in directory name (e.g., "2B", "3B", "large")
        import re
        dir_name = self.dir_model.name.lower()
        model_size = None
        
        # Try to extract size from directory name
        size_match = re.search(r'(\d+(?:\.\d+)?b)', dir_name)
        if size_match:
            model_size = size_match.group(1).upper()
        
        # Fallback: estimate from hidden_size if not in name
        if not model_size:
            hidden_size = self.hparams.get("hidden_size", 0)
            # Rough heuristic: hidden_size 1024~1500 -> 1B, 2000~3000 -> 2-3B, 4000+ -> 7B+
            if hidden_size >= 4000:
                model_size = "7B"
            elif hidden_size >= 2000:
                model_size = "3B" if hidden_size >= 2600 else "2B"
            elif hidden_size >= 1000:
                model_size = "1B"
        
        if model_size:
            logger.info(f"gguf: model size = {model_size}")

    def weight_quant(self, weight):
        """Compute quantization scale but don't quantize yet - we need the scale for saving"""
        dtype = weight.dtype
        weight = weight.float()
        s = 1 / weight.abs().mean().clamp(min=1e-5)
        return s  # Return only the scale, not the quantized weights

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # Don't quantize here - it will be done in write_tensors where we can save scales
        return [(self.map_tensor_name(name), data_torch)]

    def write_tensors(self):
        max_name_len = max(len(s) for _, s in self.tensor_map.mapping.values()) + len(".weight,")

        for name, data_torch in self.get_tensors():
            # we don't need these
            if name.endswith((".attention.masked_bias", ".attention.bias", ".rotary_emb.inv_freq")):
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            # use the first number-like part of the tensor name as the block id
            bid = None
            for part in name.split("."):
                if part.isdecimal():
                    bid = int(part)
                    break

            for new_name, data_torch in self.modify_tensors(data_torch, name, bid):
                data = data_torch.squeeze().numpy()
                data: np.ndarray = data  # type hint
                data_shape = data.shape
                shape_before_quant = data_shape
                n_dims = len(data.shape)
                data_dtype = data.dtype
                data_qtype: gguf.GGMLQuantizationType | None = None

                extra_f32 = False
                extra_f16 = False

                # Most of the codebase that takes in 1D tensors or norms only handles F32 tensors
                extra_f32 = any(cond for cond in (
                    extra_f32,
                    n_dims == 1,
                    new_name.endswith("_norm.weight"),
                ))

                # Some tensor types are always in float32
                tensors_f32 = [
                    gguf.MODEL_TENSOR.POS_EMBD,
                    gguf.MODEL_TENSOR.TOKEN_TYPES,
                ]
                extra_f32 = extra_f32 or any(self.match_model_tensor_name(new_name, key, bid) for key in tensors_f32)

                # if f16 desired, convert any float32 2-dim weight tensors to float16
                extra_f16 = any(cond for cond in (
                    extra_f16,
                    (name.endswith(".weight") and n_dims >= 2),
                ))

                suit_i2 = True
                if name.endswith('embed_tokens.weight') or name.endswith('norm.weight'):
                    suit_i2 = False

                i2_scale = None
                if self.ftype != gguf.GGMLQuantizationType.F32 and extra_f16 and not extra_f32:
                    # Convert to float32 first for quantization processing
                    data_f32 = data.astype(np.float32)
                    
                    if self.ftype == gguf.GGMLQuantizationType.TL1 and suit_i2:
                        # For ternary quantization: apply weight_quant before TL1 packing
                        data, i2_scale = transform_to_tl1(data_f32)
                        assert data.dtype == np.uint8
                        assert i2_scale.dtype in (np.float32, np.float64, float)
                        data_qtype = gguf.GGMLQuantizationType.TL1
                    elif self.ftype == gguf.GGMLQuantizationType.TL2 and suit_i2:
                        # For ternary quantization: apply weight_quant before TL2 packing
                        if name.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight", 
                                          "down_proj.weight", "up_proj.weight", "gate_proj.weight",
                                          "o_proj.weight")):
                            # Apply ternary quantization (BitNet specific)
                            s = 1.0 / (np.abs(data_f32).mean() + 1e-5)
                            data = np.round(data_f32 * s).clip(-1, 1) / s
                        data, i2_scale = transform_to_tl2(data)
                        assert data.dtype == np.uint8
                        assert i2_scale.dtype in (np.float32, np.float64, float)
                        data_qtype = gguf.GGMLQuantizationType.TL2
                    elif name.endswith('embed_tokens.weight'):  # quantize embedding layer to f16
                        abs_max = np.max(np.abs(data_f32))
                        if abs_max > 65504:
                            print(f"!!! CRITICAL: Max value {abs_max} will OVERFLOW FP16.")
                            # Count how many elements are problematic
                            overflow_count = np.sum(np.abs(data_f32) > 65504)
                            print(f"Number of overflowing elements: {overflow_count}")

                        if not np.isfinite(data_f32).all():
                            print("!!! WARNING: Data contains NaN or Inf before conversion.")
                            print(f"NaNs: {np.isnan(data_f32).sum()} | Infs: {np.isinf(data_f32).sum()}")

                        data = data_f32.astype(np.float16)
                        data_qtype = gguf.GGMLQuantizationType.F16

                if data_qtype is None:  # by default, convert to float32
                    if data_dtype != np.float32:
                        data = data.astype(np.float32)
                    data_qtype = gguf.GGMLQuantizationType.F32


                shape = data.shape
                # reverse shape to make it similar to the internal ggml dimension order
                shape_str = f"{{{', '.join(str(n) for n in reversed(shape))}}}"

                # n_dims is implicit in the shape
                logger.info(f"{f'%-{max_name_len}s' % f'{new_name},'} {old_dtype} --> {data_qtype.name}, shape = {shape_str}")

                # For TL1/TL2, pass the original logical shape so llama.cpp gets the right tensor dimensions
                raw_shape = shape_before_quant if data_qtype in (gguf.GGMLQuantizationType.TL1, gguf.GGMLQuantizationType.TL2) else data.shape
                self.gguf_writer.add_tensor(new_name, data, raw_shape=raw_shape, raw_dtype=data_qtype)
                if i2_scale is not None:
                    i2_scale = np.asarray(i2_scale, dtype=np.float32)
                    # Ensure scale is 1D array with at least one element for GGUF compatibility
                    if i2_scale.ndim == 0:
                        i2_scale = np.array([i2_scale.item()], dtype=np.float32)
                    self.gguf_writer.add_tensor(new_name + "_scale", i2_scale, raw_dtype=gguf.GGMLQuantizationType.F32)
                    logger.info(f"    Added scale tensor: {new_name + '_scale'}, F32, shape={i2_scale.shape}, value: {i2_scale.item()}")




###### CONVERSION LOGIC ######


ftype_map = {
    "f32": gguf.GGMLQuantizationType.F32,
    "f16": gguf.GGMLQuantizationType.F16,
    "tl1" : gguf.GGMLQuantizationType.TL1,
    "tl2" : gguf.GGMLQuantizationType.TL2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a huggingface model to a GGML compatible file")
    parser.add_argument(
        "--vocab-only", action="store_true",
        help="extract only the vocab",
    )
    parser.add_argument(
        "--awq-path", type=Path, default=None,
        help="Path to scale awq cache file")
    parser.add_argument(
        "--outfile", type=Path,
        help="path to write to; default: based on input",
    )
    parser.add_argument(
        "--outtype", type=str, choices=ftype_map.keys(), default="f32",
        help="output format - use f32 for float32, f16 for float16",
    )
    parser.add_argument("--bigendian", action="store_true", help="model is executed on big endian machine")
    parser.add_argument(
        "model", type=Path,
        help="directory containing model file",
    )
    parser.add_argument("--use-temp-file", action="store_true", help="use the tempfile library while processing (helpful when running out of memory, process killed)")
    parser.add_argument("--model-name", type=str, default=None, help="name of the model")
    parser.add_argument("--verbose", action="store_true", help="increase output verbosity")

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    dir_model = args.model

    if not dir_model.is_dir():
        logger.error(f'Error: {args.model} is not a directory')
        sys.exit(1)

    if args.outfile is not None:
        fname_out = args.outfile
    else:
        # output in the same directory as the model by default
        fname_out = dir_model / f'ggml-model-{args.outtype}.gguf'

    logger.info(f"Loading model: {dir_model.name}")

    hparams = Model.load_hparams(dir_model)

    with torch.inference_mode():
        model_class = Model.from_model_architecture(hparams["architectures"][0])
        model_instance = model_class(dir_model, ftype_map[args.outtype], fname_out, args.bigendian, args.use_temp_file)

        logger.info("Set model parameters")
        model_instance.set_gguf_parameters()

        logger.info("Set model tokenizer")
        model_instance.set_vocab()

        if args.vocab_only:
            logger.info(f"Exporting model vocab to '{fname_out}'")
            model_instance.write_vocab()
        else:
            logger.info(f"Exporting model to '{fname_out}'")
            model_instance.write()

        logger.info(f"Model successfully exported to '{fname_out}'")


if __name__ == '__main__':
    args = parse_args()

    main()
