"""Safetensors-backed storage для CaptureData.

Layout (внутри файла):
  q_l{ℓ}, k_pre_l{ℓ}, v_pre_l{ℓ}, k_post_l{ℓ}, v_post_l{ℓ}  — per layer
  logits

Meta хранится в safetensors `__metadata__` как JSON-string (safetensors
требует str→str). Поэтому при load распарсиваем обратно из JSON.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


@dataclass
class CaptureData:
    meta: dict[str, Any]
    q: list[torch.Tensor]            # per layer, pre-RoPE [W, hidden]
    k_pre: list[torch.Tensor]
    v_pre: list[torch.Tensor]
    k_post: list[torch.Tensor]
    v_post: list[torch.Tensor]
    logits: torch.Tensor
    # Опциональный post-RoPE Q [W, num_heads, head_dim].
    # None для старых captures без apply_rotary_pos_emb hook'а.
    q_post_rope: list[torch.Tensor] | None = None


def save_capture(cap: CaptureData, path: Path) -> None:
    n_layers = len(cap.q)
    for field_name, tensors_list in [
        ("k_pre", cap.k_pre),
        ("v_pre", cap.v_pre),
        ("k_post", cap.k_post),
        ("v_post", cap.v_post),
    ]:
        if len(tensors_list) != n_layers:
            raise ValueError(
                f"Layer count mismatch: q has {n_layers} layers, "
                f"{field_name} has {len(tensors_list)}"
            )
    has_q_post_rope = cap.q_post_rope is not None and len(cap.q_post_rope) == n_layers
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    for layer in range(n_layers):
        tensors[f"q_l{layer}"] = cap.q[layer].contiguous()
        tensors[f"k_pre_l{layer}"] = cap.k_pre[layer].contiguous()
        tensors[f"v_pre_l{layer}"] = cap.v_pre[layer].contiguous()
        tensors[f"k_post_l{layer}"] = cap.k_post[layer].contiguous()
        tensors[f"v_post_l{layer}"] = cap.v_post[layer].contiguous()
        if has_q_post_rope:
            tensors[f"q_post_rope_l{layer}"] = cap.q_post_rope[layer].contiguous()
    tensors["logits"] = cap.logits.contiguous()

    meta_with_layers = {**cap.meta, "n_layers": n_layers,
                        "has_q_post_rope": has_q_post_rope}
    metadata_json = {"json": json.dumps(meta_with_layers)}
    save_file(tensors, str(path), metadata=metadata_json)


def load_capture(path: Path) -> CaptureData:
    # safetensors split — load tensors first, then header metadata via the
    # safe_open context. Simpler: load all + re-parse header.
    raw = load_file(str(path))

    # Read metadata via low-level API.
    from safetensors import safe_open
    with safe_open(str(path), framework="pt") as f:
        meta_str = f.metadata().get("json")
    if meta_str is None:
        raise ValueError(f"Capture file {path} missing 'json' metadata key")
    meta = json.loads(meta_str)
    n_layers = meta.pop("n_layers")
    has_q_post_rope = meta.pop("has_q_post_rope", False)

    q = [raw[f"q_l{ℓ}"] for ℓ in range(n_layers)]
    k_pre = [raw[f"k_pre_l{ℓ}"] for ℓ in range(n_layers)]
    v_pre = [raw[f"v_pre_l{ℓ}"] for ℓ in range(n_layers)]
    k_post = [raw[f"k_post_l{ℓ}"] for ℓ in range(n_layers)]
    v_post = [raw[f"v_post_l{ℓ}"] for ℓ in range(n_layers)]
    logits = raw["logits"]
    q_post_rope = None
    if has_q_post_rope:
        q_post_rope = [raw[f"q_post_rope_l{ℓ}"] for ℓ in range(n_layers)]

    return CaptureData(meta=meta, q=q, k_pre=k_pre, v_pre=v_pre,
                       k_post=k_post, v_post=v_post, logits=logits,
                       q_post_rope=q_post_rope)
