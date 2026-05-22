"""Capture pipeline orchestrator на CPU.

Загружает Qwen3-1.7B в bf16 на CPU, запускает forward с hook'ами
для teacher-forced или autoregressive захвата.
"""
from __future__ import annotations

import datetime
import logging
from typing import Any

import torch

from kvtrace.capture.attention_hooks import install_capture_hooks
from kvtrace.capture.fp8_sim import QUANT_FNS
from kvtrace.capture.storage import CaptureData
from kvtrace.capture.window import Window

log = logging.getLogger(__name__)


class CaptureRunner:
    """Single-model lifecycle: load → many captures → unload."""

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._attention_modules: list[Any] = []
        self._n_layers: int = 0
        self._vocab: int = 0
        self._model_revision_hash: str = ""

    def load_model(self, hf_id: str, trust_remote_code: bool = True) -> None:
        """Load Qwen3-1.7B in bf16 on CPU with eager attention."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=trust_remote_code)
        self._model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=torch.bfloat16,
            device_map={"": "cpu"},
            trust_remote_code=trust_remote_code,
            attn_implementation="eager",
        )
        self._model.eval()
        # Find attention modules. Qwen3: model.model.layers[i].self_attn
        layers = self._model.model.layers
        self._attention_modules = [layer.self_attn for layer in layers]
        self._n_layers = len(self._attention_modules)
        self._vocab = self._model.config.vocab_size
        self._model_revision_hash = getattr(self._model.config, "_commit_hash", "unknown")

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._attention_modules = []

    def run_tf(
        self,
        input_token_ids: list[int],
        window: Window,
        quant: str,
        problem_id: int,
        fdp_token_idx: int,
    ) -> CaptureData:
        """One forward pass on input_token_ids[0:window.we], slice to [ws:we]."""
        assert self._model is not None, "call load_model() first"
        if quant not in QUANT_FNS:
            raise ValueError(f"Unknown quant {quant!r}; expected one of {list(QUANT_FNS)}")
        quant_fn = QUANT_FNS[quant]

        handle = install_capture_hooks(
            self._model,
            attention_modules=self._attention_modules,
            quant_fn=quant_fn,
        )
        try:
            # Feed tokens [0:we]
            feed_ids = torch.tensor([input_token_ids[: window.we]], dtype=torch.long)
            with torch.no_grad():
                out = self._model(input_ids=feed_ids, use_cache=True)
            logits_full = out.logits[0]  # [we, vocab]

            ws, we = window.ws, window.we
            q_sliced = [_slice_q(t, ws, we) for t in handle.q]
            q_post_rope_sliced = (
                [_slice_kv(t, ws, we) for t in handle.q_post_rope]
                if handle.q_post_rope else None
            )
            k_pre_sliced = [_slice_kv(t, ws, we) for t in handle.k_pre]
            v_pre_sliced = [_slice_kv(t, ws, we) for t in handle.v_pre]
            k_post_sliced = [_slice_kv(t, ws, we) for t in handle.k_post]
            v_post_sliced = [_slice_kv(t, ws, we) for t in handle.v_post]
            logits_sliced = logits_full[ws:we]

            return _build_capture(
                quant=quant,
                mode="tf",
                q_post_rope=q_post_rope_sliced,
                problem_id=problem_id,
                fdp_token_idx=fdp_token_idx,
                window=window,
                input_token_ids=input_token_ids[ws:we],
                gen_token_ids=input_token_ids[ws:we],
                early_eos=False,
                model_revision_hash=self._model_revision_hash,
                q=q_sliced,
                k_pre=k_pre_sliced,
                v_pre=v_pre_sliced,
                k_post=k_post_sliced,
                v_post=v_post_sliced,
                logits=logits_sliced,
            )
        finally:
            handle.remove()

    def run_ar(
        self,
        prefix_token_ids: list[int],
        window: Window,
        quant: str,
        problem_id: int,
        fdp_token_idx: int,
        max_new_tokens: int = 250,
        repetition_penalty: float = 1.05,
    ) -> CaptureData:
        """Prefill prefix → greedy decode max_new_tokens с KV-quant в loop."""
        assert self._model is not None, "call load_model() first"
        if quant not in QUANT_FNS:
            raise ValueError(f"Unknown quant {quant!r}")
        quant_fn = QUANT_FNS[quant]

        handle = install_capture_hooks(
            self._model,
            attention_modules=self._attention_modules,
            quant_fn=quant_fn,
        )
        try:
            input_ids = torch.tensor([prefix_token_ids], dtype=torch.long)
            with torch.no_grad():
                out = self._model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,  # ignored when do_sample=False
                    repetition_penalty=repetition_penalty,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            full_sequence: list[int] = out.sequences[0].tolist()
            # Logits per generated step: out.scores is tuple of [B, vocab]
            gen_logits = torch.stack(list(out.scores), dim=0)  # [n_gen, B, vocab] or [n_gen, vocab]
            if gen_logits.dim() == 3:
                gen_logits = gen_logits[:, 0, :]
            n_generated = gen_logits.shape[0]
            early_eos = n_generated < max_new_tokens

            # Build aligned tensor sequence: hooks collected per-call.
            # In append mode, handle.q[i::n_layers] gives all captures for
            # layer i across all generate steps (prefill + each decode step).
            q_all = [_concat_seq(handle.q[i::self._n_layers]) for i in range(self._n_layers)]
            k_pre_all = [_concat_seq(handle.k_pre[i::self._n_layers]) for i in range(self._n_layers)]
            v_pre_all = [_concat_seq(handle.v_pre[i::self._n_layers]) for i in range(self._n_layers)]
            k_post_all = [_concat_seq(handle.k_post[i::self._n_layers]) for i in range(self._n_layers)]
            v_post_all = [_concat_seq(handle.v_post[i::self._n_layers]) for i in range(self._n_layers)]
            q_post_rope_all = (
                [_concat_seq(handle.q_post_rope[i::self._n_layers])
                 for i in range(self._n_layers)]
                if handle.q_post_rope else None
            )

            ws, we = window.ws, window.we
            q_sliced = [_slice_q(t, ws, we) for t in q_all]
            q_post_rope_sliced = (
                [_slice_kv(t, ws, we) for t in q_post_rope_all]
                if q_post_rope_all else None
            )
            k_pre_sliced = [_slice_kv(t, ws, we) for t in k_pre_all]
            v_pre_sliced = [_slice_kv(t, ws, we) for t in v_pre_all]
            k_post_sliced = [_slice_kv(t, ws, we) for t in k_post_all]
            v_post_sliced = [_slice_kv(t, ws, we) for t in v_post_all]

            # Logits: only generated positions are available from generate().
            # Fill prefix positions with zeros; concat with actual generated logits.
            prefix_len = len(prefix_token_ids)
            full_logits = torch.zeros(prefix_len + n_generated, self._vocab, dtype=gen_logits.dtype)
            full_logits[prefix_len : prefix_len + n_generated] = gen_logits
            logits_sliced = full_logits[ws:we]

            return _build_capture(
                quant=quant,
                mode="ar",
                q_post_rope=q_post_rope_sliced,
                problem_id=problem_id,
                fdp_token_idx=fdp_token_idx,
                window=window,
                input_token_ids=prefix_token_ids + [0] * n_generated,
                gen_token_ids=full_sequence[ws:we],
                early_eos=early_eos,
                model_revision_hash=self._model_revision_hash,
                q=q_sliced,
                k_pre=k_pre_sliced,
                v_pre=v_pre_sliced,
                k_post=k_post_sliced,
                v_post=v_post_sliced,
                logits=logits_sliced,
            )
        finally:
            handle.remove()


def _slice_q(t: torch.Tensor | None, ws: int, we: int) -> torch.Tensor:
    """Q layout: [B=1, seq, hidden] (from q_proj) or [seq, hidden] (after AR concat).

    Returns [W, hidden]. User can reshape to [W, num_heads, head_dim] offline.
    Note: Q is captured pre-RoPE (q_proj output); reshape correctness depends on
    that interpretation.
    """
    if t is None:
        return torch.empty(0, dtype=torch.float16)
    if t.dim() == 3 and t.shape[0] == 1:
        return t[0, ws:we].to(torch.float16).contiguous()
    if t.dim() == 2:
        return t[ws:we].to(torch.float16).contiguous()
    return t


def _slice_kv(t: torch.Tensor | None, ws: int, we: int) -> torch.Tensor:
    """K/V layout: [B=1, num_kv_heads, seq, head_dim] (TF) or
    [num_kv_heads, seq, head_dim] (AR after _concat_seq took last).

    Returns [W, num_kv_heads, head_dim] — seq becomes dim 0 to match Q convention
    and match the spec layout.
    """
    if t is None:
        return torch.empty(0, dtype=torch.float16)
    if t.dim() == 4 and t.shape[0] == 1:
        return t[0, :, ws:we, :].permute(1, 0, 2).to(torch.float16).contiguous()
    if t.dim() == 3:
        return t[:, ws:we, :].permute(1, 0, 2).to(torch.float16).contiguous()
    return t


def _slice_seq_dim(t: torch.Tensor | None, ws: int, we: int) -> torch.Tensor:
    """DEPRECATED: kept for backwards-compat. Use _slice_q or _slice_kv directly.

    Heuristic: 4D → K/V layout; 3D with batch=1 → Q layout.
    """
    if t is None:
        return torch.empty(0, dtype=torch.float16)
    if t.dim() == 4:
        return _slice_kv(t, ws, we)
    if t.dim() >= 3 and t.shape[0] == 1:
        return t[0, ws:we].to(torch.float16).contiguous()
    if t.dim() >= 2:
        return t[ws:we].to(torch.float16).contiguous()
    return t


def _concat_seq(tensor_list: list[torch.Tensor]) -> torch.Tensor:
    """Concat per-call captures along seq dim.

    Both Q ([B, seq, hidden]) and K/V ([B, num_kv_heads, seq, head_dim])
    have seq at dim=-2, so torch.cat(dim=-2) handles both uniformly.

    After the AR hook fix (capture only new positions per call), tensors
    in the list are non-overlapping slices that compose into the full
    sequence — no more cumulative-cache shape mismatches.
    """
    if not tensor_list:
        return torch.empty(0)
    try:
        return torch.cat(tensor_list, dim=-2)
    except RuntimeError:
        # Legacy fallback for shape mismatches: pick largest.
        return max(tensor_list, key=lambda t: t.numel())


def _build_capture(
    *,
    quant: str,
    mode: str,
    problem_id: int,
    fdp_token_idx: int,
    window: Window,
    input_token_ids: list[int],
    gen_token_ids: list[int],
    early_eos: bool,
    model_revision_hash: str,
    q: list[torch.Tensor],
    k_pre: list[torch.Tensor],
    q_post_rope: list[torch.Tensor] | None = None,
    v_pre: list[torch.Tensor],
    k_post: list[torch.Tensor],
    v_post: list[torch.Tensor],
    logits: torch.Tensor,
) -> CaptureData:
    import transformers

    meta = {
        "model": "qwen3-1.7b",
        "quant": quant,
        "mode": mode,
        "problem_id": problem_id,
        "fdp_token_idx": fdp_token_idx,
        "window_start": window.ws,
        "window_end": window.we,
        "W": window.size,
        "input_token_ids": list(input_token_ids),
        "gen_token_ids": list(gen_token_ids),
        "truncated_left": window.truncated_left,
        "truncated_right": window.truncated_right,
        "early_eos": early_eos,
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "model_revision_hash": model_revision_hash,
        "run_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }
    return CaptureData(
        meta=meta,
        q=q,
        k_pre=k_pre,
        v_pre=v_pre,
        k_post=k_post,
        v_post=v_post,
        logits=logits.to(torch.float16).contiguous(),
        q_post_rope=q_post_rope,
    )
