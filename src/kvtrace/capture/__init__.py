"""KV-cache capture pipeline (Phase 6).

Captures Q, K_pre, K_post, V_pre, V_post, logits during forward pass on CPU
under bf16 / fp8_e4m3 / fp8_e5m2 KV-cache quantization. See
docs/superpowers/specs/2026-05-21-kv-matrix-capture-design.md.
"""
from __future__ import annotations
