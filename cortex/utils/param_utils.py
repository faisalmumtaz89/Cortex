"""Lightweight parameter detection and formatting utilities (no heavy deps)."""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Optional, Dict, Any


def _is_parameter_tensor(tensor_name: str) -> bool:
    # Common parameter patterns
    param_patterns = [
        'weight', 'bias', 'embeddings', 'lm_head',
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj',
        'fc1', 'fc2', 'mlp', 'attention'
    ]
    # Common non-parameter patterns (buffers)
    non_param_patterns = [
        'position_ids', 'attention_mask', 'token_type_ids',
        'freqs_cos', 'freqs_sin', 'inv_freq'
    ]
    name = tensor_name.lower()
    if any(p in name for p in non_param_patterns):
        return False
    if any(p in name for p in param_patterns):
        return True
    return any(ind in name for ind in ['layer', 'block', 'transformer'])


def _read_safetensors_header(file_path: Path) -> Optional[int]:
    try:
        with open(file_path, 'rb') as f:
            header_size_bytes = f.read(8)
            if len(header_size_bytes) < 8:
                return None
            header_size = struct.unpack('<Q', header_size_bytes)[0]
            header_json = f.read(header_size).decode('utf-8')
            header = json.loads(header_json)
            total_params = 0
            for tensor_name, tensor_info in header.items():
                if tensor_name == "__metadata__":
                    continue
                if _is_parameter_tensor(tensor_name):
                    shape = tensor_info.get('shape', [])
                    if shape:
                        n = 1
                        for d in shape:
                            n *= d
                        total_params += n
            return total_params
    except Exception:
        return None


def _detect_config_parameters(model_path: Path) -> Optional[int]:
    try:
        cfg_path = model_path / "config.json"
        if not cfg_path.exists():
            return None
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)
        if 'num_parameters' in cfg:
            return int(cfg['num_parameters'])
        # Architecture-based calculation (subset): generic transformer fallback
        model_type = cfg.get('model_type', '').lower()
        if model_type in ['llama', 'gemma', 'mistral', 'qwen']:
            return _calc_llama_params(cfg)
        if model_type in ['gpt', 'gpt2', 'gpt_neo', 'gpt_neox']:
            return _calc_gpt_params(cfg)
        if model_type in ['bert', 'roberta', 'distilbert']:
            return _calc_bert_params(cfg)
        return _calc_generic_transformer_params(cfg)
    except Exception:
        return None


def _calc_llama_params(cfg: Dict[str, Any]) -> Optional[int]:
    try:
        vocab_size = cfg.get('vocab_size', 32000)
        hidden_size = cfg.get('hidden_size', 4096)
        intermediate_size = cfg.get('intermediate_size', 11008)
        num_layers = cfg.get('num_hidden_layers', 32)
        num_attention_heads = cfg.get('num_attention_heads', 32)
        is_gemma = cfg.get('model_type', '').lower() == 'gemma'
        embedding_params = vocab_size * hidden_size
        if is_gemma:
            num_kv = cfg.get('num_key_value_heads', max(1, num_attention_heads // 4))
            head_dim = hidden_size // num_attention_heads
            q_proj = hidden_size * hidden_size
            k_proj = hidden_size * (num_kv * head_dim)
            v_proj = hidden_size * (num_kv * head_dim)
            o_proj = hidden_size * hidden_size
            attention = q_proj + k_proj + v_proj + o_proj
        else:
            attention = 4 * (hidden_size * hidden_size)
        ff = 2 * (hidden_size * intermediate_size) + (intermediate_size * hidden_size)
        ln = 2 * hidden_size
        layer = attention + ff + ln
        transformer = num_layers * layer
        final_ln = hidden_size
        tie_word_embeddings = cfg.get('tie_word_embeddings', True)
        lm_head = 0 if tie_word_embeddings else vocab_size * hidden_size
        return embedding_params + transformer + final_ln + lm_head
    except Exception:
        return None


def _calc_gpt_params(cfg: Dict[str, Any]) -> Optional[int]:
    try:
        vocab_size = cfg.get('vocab_size', 50257)
        n_embd = cfg.get('n_embd', cfg.get('hidden_size', 768))
        n_layer = cfg.get('n_layer', cfg.get('num_hidden_layers', 12))
        n_head = cfg.get('n_head', cfg.get('num_attention_heads', 12))
        max_pos = cfg.get('n_positions', cfg.get('max_position_embeddings', 1024))
        embedding = vocab_size * n_embd + max_pos * n_embd
        attention = 4 * (n_embd * n_embd)
        mlp_size = cfg.get('n_inner', 4 * n_embd)
        mlp = n_embd * mlp_size + mlp_size * n_embd
        ln = 2 * n_embd
        block = attention + mlp + ln
        transformer = n_layer * block
        final_ln = n_embd
        lm_head = vocab_size * n_embd
        return embedding + transformer + final_ln + lm_head
    except Exception:
        return None


def _calc_bert_params(cfg: Dict[str, Any]) -> Optional[int]:
    try:
        vocab_size = cfg.get('vocab_size', 30522)
        hidden = cfg.get('hidden_size', 768)
        layers = cfg.get('num_hidden_layers', 12)
        intermediate = cfg.get('intermediate_size', 3072)
        max_pos = cfg.get('max_position_embeddings', 512)
        type_vocab = cfg.get('type_vocab_size', 2)
        embedding = vocab_size * hidden + max_pos * hidden + type_vocab * hidden
        attention = 4 * (hidden * hidden)
        ff = hidden * intermediate + intermediate * hidden
        ln = 2 * hidden
        layer = attention + ff + ln
        encoder = layers * layer
        pooler = hidden * hidden
        return embedding + encoder + pooler
    except Exception:
        return None


def _calc_generic_transformer_params(cfg: Dict[str, Any]) -> Optional[int]:
    try:
        vocab_size = cfg.get('vocab_size', 32000)
        hidden_size = cfg.get('hidden_size', cfg.get('n_embd', cfg.get('d_model', 512)))
        num_layers = cfg.get('num_hidden_layers', cfg.get('n_layer', cfg.get('num_layers', 6)))
        if hidden_size is None or num_layers is None:
            return None
        embedding = vocab_size * hidden_size
        layer = 6 * (hidden_size * hidden_size)
        transformer = num_layers * layer
        head = vocab_size * hidden_size
        return embedding + transformer + head
    except Exception:
        return None


def detect_model_parameters(model_path: Path) -> Optional[int]:
    """Detect parameter count using light-weight methods (safetensors, config)."""
    try:
        if not model_path.exists():
            return None
        if model_path.is_dir():
            # Prefer safetensors header-based detection
            st_files = list(model_path.glob('*.safetensors'))
            if st_files:
                total = 0
                for st in st_files:
                    if 'adapter' in st.name.lower():
                        continue
                    params = _read_safetensors_header(st)
                    if params is None:
                        return None
                    total += params
                return total if total > 0 else None
            # Fallback to config.json
            cfg_params = _detect_config_parameters(model_path)
            if cfg_params is not None:
                return cfg_params
        # Not recognized
        return None
    except Exception:
        return None


def _estimate_weight_file_size(model_path: Path) -> int:
    if model_path.is_file():
        return model_path.stat().st_size
    weight_patterns = ['*.safetensors', '*.bin', '*.npz', 'pytorch_model*.bin', 'model*.safetensors']
    non_weight_patterns = [
        'tokenizer*', 'vocab*', 'merges.txt', 'config.json',
        'generation_config.json', 'special_tokens_map.json',
        'tokenizer_config.json', 'added_tokens.json'
    ]
    total = 0
    for p in model_path.rglob('*'):
        if p.is_file():
            is_weight = any(p.match(pattern) for pattern in weight_patterns)
            is_non_weight = any(p.match(pattern) for pattern in non_weight_patterns)
            if is_weight and not is_non_weight:
                total += p.stat().st_size
            elif not is_non_weight and p.suffix in ['.safetensors', '.bin', '.npz']:
                total += p.stat().st_size
    return total


def get_model_parameters_smart(model_path: Path) -> float:
    """Return parameter count in billions (B)."""
    pc = detect_model_parameters(model_path)
    if pc is not None:
        return pc / 1e9
    # fallback to size-based using weight-only size
    weight_size = _estimate_weight_file_size(model_path)
    size_gb = weight_size / (1024**3)
    if size_gb <= 0:
        size_gb = 0.0
    # Better default: ~2.2 GB per 1B params
    return size_gb / 2.2


def format_param_count(params_b: Optional[float]) -> str:
    try:
        if params_b is None:
            return "unknown"
        if params_b >= 1000:
            return f"{params_b / 1000:.1f}T"
        if params_b >= 1:
            return f"{params_b:.1f}B"
        if params_b >= 0.01:
            return f"{params_b * 1000:.0f}M"
        if params_b >= 0.001:
            return f"{params_b * 1000:.1f}M"
        if params_b > 0:
            return f"{params_b * 1e6:.0f}K"
        return "0"
    except Exception:
        try:
            return f"{float(params_b):.2f}B"
        except Exception:
            return "unknown"
